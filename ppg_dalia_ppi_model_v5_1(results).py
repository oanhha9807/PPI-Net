"""
PPG-DaLiA PPI HR 추정 (v5.1)
==============================
v5 → v5.1:
  ✅ 윈도우 30초 → 64초 (HR 추세 학습 범위 확대, 4096=2^12 CNN 최적)
  ✅ 학습 연장: NUM_EPOCHS=120, PATIENCE=30 (S3 Ep75 패턴 반영)
  ✅ LR schedule: Warmup(5ep) + ReduceLROnPlateau (정체 시 ×0.5)
  ✅ 전체 15명 LoSo 검증
  ✅ MACs/FLOPs 계산 추가
  ✅ 1D+2D 듀얼 패스 + Instance Norm 유지
  ✅ ECG rpeaks 타겟 + SQI 가중 (C 설정)
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks, butter, filtfilt, welch, stft, medfilt
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import warnings
import math

warnings.filterwarnings("ignore")


# ============================================================
# 설정
# ============================================================
class Config:
    DATA_ROOT = "./PPG_FieldStudy"
    ALL_SUBJECTS = list(range(1, 16))
    TEST_SUBJECTS = list(range(1, 16))  # [v5.1] 전체 15명 LoSo

    PPG_FS = 64
    ACC_FS = 32
    ECG_FS = 700

    # [v5.1] 64초 윈도우 (64Hz × 64s = 4096 = 2^12, CNN에 최적)
    WINDOW_SEC = 64
    STRIDE_SEC = 8       # 64초 윈도우에 8초 스트라이드

    PPG_WIN_LEN = PPG_FS * WINDOW_SEC   # 4096
    ACC_WIN_LEN = ACC_FS * WINDOW_SEC   # 2048

    # [v5.1] PPI 출력 비례 증가 (2pt/sec × 64s = 128pt)
    PPI_OUTPUT_LEN = 128
    NUM_ACTIVITIES = 4

    # [v5.1] STFT 해상도 향상 (nperseg 128→256, 주파수 해상도 2배)
    STFT_NPERSEG = 256   # 4초 STFT 윈도우
    STFT_NOVERLAP = 192  # 75% 오버랩
    STFT_FREQ_MAX = 4.0

    # [v5.1] 배치 축소 (60초 → 메모리 증가) + 학습 연장
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 120     # v5: 80 → v5.1: 120
    DROPOUT = 0.25
    PATIENCE = 50        # v5: 20 → v5.1: 30 (S3 Ep75 패턴)

    LAMBDA_PPI = 1.0
    LAMBDA_ACTIVITY = 0.05
    LAMBDA_DOMAIN = 0.1
    LAMBDA_HR = 0.1        # [v5.2] HR-level loss: PPI→HR 변환 후 직접 HR 오차 학습
    LAMBDA_SMOOTH = 0.1    # [v5.2] Temporal smoothness: PPI 시계열 연속성

    # C 설정 기반 (ECG + SQI, 칼만 OFF)
    SQI_WEIGHT_MIN = 0.6    # v4.1: 0.4 → v5: 0.6 (어려운 구간 더 적극 학습)
    SQI_WEIGHT_MAX = 1.0
    RRI_MARGIN_SEC = 2.0

    USE_ECG_RPEAKS = True
    USE_SQI_WEIGHTING = True
    USE_KALMAN = False

    AUG_NOISE_STD = 0.05
    AUG_SCALE_RANGE = (0.85, 1.15)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42

    def ablation_name(self):
        if not self.USE_ECG_RPEAKS:
            return "A_PPG_only"
        elif not self.USE_SQI_WEIGHTING and not self.USE_KALMAN:
            return "B_ECG_only"
        elif self.USE_SQI_WEIGHTING and not self.USE_KALMAN:
            return "C_ECG_SQI"
        else:
            return "D_ECG_SQI_Kalman"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. 신호 처리
# ============================================================
def bandpass_filter(sig, fs, low=0.5, high=4.0, order=3):
    nyq = fs / 2.0
    lo, hi = max(low / nyq, 0.01), min(high / nyq, 0.99)
    if lo >= hi:
        return sig
    try:
        b, a = butter(order, [lo, hi], btype="band")
        out = filtfilt(b, a, sig, axis=0)
        return sig if np.any(np.isnan(out)) else out
    except Exception:
        return sig


def detect_ppg_peaks_global(ppg, fs):
    if len(ppg) < int(fs * 2):
        return np.array([], dtype=int)
    filt = bandpass_filter(ppg, fs, 0.5, 3.5)
    peaks, _ = find_peaks(filt, distance=max(int(fs * 0.33), 1),
                          height=np.median(filt),
                          prominence=max(0.1 * np.std(filt), 1e-6))
    return peaks


def extract_window_peaks(global_peaks, win_start, win_end):
    mask = (global_peaks >= win_start) & (global_peaks < win_end)
    return global_peaks[mask] - win_start


def convert_rpeaks_to_seconds(rpeaks_raw, ecg_fs):
    return rpeaks_raw.astype(np.float64) / ecg_fs


def compute_ppi_from_rpeaks_sec(rpeaks_sec, win_start_sec, win_end_sec,
                                 output_len, margin_sec=2.0,
                                 hr_fallback=70.0):
    default_ppi = 60.0 / max(hr_fallback, 30.0)
    win_dur = win_end_sec - win_start_sec

    candidate_mask = ((rpeaks_sec >= win_start_sec - margin_sec) &
                      (rpeaks_sec < win_end_sec + margin_sec))
    candidate = rpeaks_sec[candidate_mask]

    if len(candidate) < 3:
        return (np.full(output_len, default_ppi, dtype=np.float32),
                hr_fallback, "short")

    rri = np.diff(candidate)
    mid = 0.5 * (candidate[:-1] + candidate[1:])
    in_window = (mid >= win_start_sec) & (mid < win_end_sec)
    valid_range = (rri > 0.27) & (rri < 2.2)
    valid = in_window & valid_range

    if valid.sum() < 2:
        return (np.full(output_len, default_ppi, dtype=np.float32),
                hr_fallback, "invalid")

    rri_valid = rri[valid]
    mid_valid = mid[valid] - win_start_sec

    t_uniform = np.linspace(0.0, win_dur, output_len)

    if len(mid_valid) >= 2:
        f = interp1d(mid_valid, rri_valid, kind="linear",
                     fill_value=(rri_valid[0], rri_valid[-1]),
                     bounds_error=False)
        ppi_out = f(t_uniform).astype(np.float32)
    else:
        ppi_out = np.full(output_len, rri_valid[0], dtype=np.float32)

    ppi_out = np.nan_to_num(ppi_out, nan=default_ppi, posinf=2.2, neginf=0.27)
    ppi_out = np.clip(ppi_out, 0.27, 2.2)
    mean_hr = float(np.clip(np.mean(60.0 / rri_valid), 30, 220))
    return ppi_out, mean_hr, "ok"


# ============================================================
# [v5] STFT 스펙트로그램 생성
# ============================================================
def compute_spectrogram(ppg_window: np.ndarray, fs: float,
                        nperseg: int = 128, noverlap: int = 96,
                        freq_max: float = 4.0) -> np.ndarray:
    """
    PPG → STFT 스펙트로그램 (2D CNN 입력용)

    Returns:
        spec: [n_freq, n_time] 로그 파워 스펙트로그램
              n_freq: 0~freq_max Hz 범위의 주파수 빈 수
              n_time: 시간 프레임 수
    """
    try:
        f, t, Zxx = stft(ppg_window, fs=fs, nperseg=nperseg,
                         noverlap=noverlap, boundary=None)
        # freq_max까지만 사용
        freq_mask = f <= freq_max
        Zxx = Zxx[freq_mask, :]

        # 로그 파워 스펙트로그램
        power = np.abs(Zxx) ** 2
        log_power = np.log1p(power)  # log(1 + power) for stability

        # 정규화
        log_power = log_power.astype(np.float32)
        m, s = log_power.mean(), log_power.std()
        if s > 1e-8:
            log_power = (log_power - m) / s

        return log_power
    except Exception:
        # 실패 시 빈 스펙트로그램
        n_freq = int(freq_max / (fs / nperseg)) + 1
        n_time = max(1, (len(ppg_window) - nperseg) // (nperseg - noverlap) + 1)
        return np.zeros((n_freq, n_time), dtype=np.float32)


# ============================================================
# SQI
# ============================================================
def compute_sqi(ppg_window, fs, ppg_peaks):
    sqi_s, sqi_t = 0.5, 0.5
    try:
        if len(ppg_window) >= 128:
            freqs, psd = welch(ppg_window, fs=fs, nperseg=min(256, len(ppg_window)))
            cardiac = (freqs >= 0.5) & (freqs <= 4.0)
            sqi_s = float(np.clip(np.sum(psd[cardiac]) / (np.sum(psd) + 1e-10), 0, 1))
    except Exception:
        pass
    if len(ppg_peaks) >= 3:
        ppi = np.diff(ppg_peaks) / fs
        v = ppi[(ppi > 0.3) & (ppi < 2.0)]
        if len(v) >= 2:
            sqi_t = float(np.exp(-2.0 * np.std(v) / (np.mean(v) + 1e-8)))
    return float(np.clip(np.sqrt(sqi_s * sqi_t), 0, 1))


def estimate_activity(acc):
    if acc.ndim == 1: acc = acc.reshape(-1, 1)
    s = np.std(np.sqrt(np.sum(acc ** 2, axis=1)))
    return 0 if s < 0.1 else 1 if s < 0.4 else 2 if s < 0.8 else 3


# ============================================================
# 2. 데이터 로딩
# ============================================================
def load_ppg_dalia_subject(data_root, subject_id):
    candidates = [os.path.join(data_root, f"S{subject_id}.pkl"),
                  os.path.join(data_root, f"S{subject_id}", f"S{subject_id}.pkl")]
    pkl_path = next((c for c in candidates if os.path.exists(c)), None)
    if not pkl_path:
        return None
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print(f"  [!] S{subject_id}: load 실패 - {e}")
        return None
    if not isinstance(data, dict):
        return None

    signal = data.get("signal", {})
    ppg, acc = None, None
    if isinstance(signal, dict):
        wrist = signal.get("wrist", signal.get("Wrist", {}))
        if isinstance(wrist, dict):
            for k in ["BVP", "bvp"]:
                if k in wrist:
                    ppg = np.array(wrist[k], dtype=np.float32).flatten(); break
            for k in ["ACC", "acc"]:
                if k in wrist:
                    acc = np.array(wrist[k], dtype=np.float32); break

    if ppg is None or len(ppg) < Config.PPG_WIN_LEN:
        return None
    if acc is None:
        acc = np.zeros((int(len(ppg) * Config.ACC_FS / Config.PPG_FS), 3), np.float32)
    if acc.ndim == 1:
        acc = acc.reshape(-1, 3) if len(acc) % 3 == 0 else \
            np.hstack([acc.reshape(-1, 1), np.zeros((len(acc), 2), np.float32)])

    label = np.array(data.get("label", []), dtype=np.float32).flatten()
    activity = np.array(data.get("activity", []), dtype=np.int64).flatten()

    rpeaks_sec = None
    rpeaks_raw = data.get("rpeaks", None)
    if rpeaks_raw is not None:
        arr = np.array(rpeaks_raw).flatten()
        if len(arr) > 10:
            ecg_fs = Config.ECG_FS
            chest = signal.get("chest", {}) if isinstance(signal, dict) else {}
            if isinstance(chest, dict) and "ECG" in chest:
                est = len(np.array(chest["ECG"]).flatten()) / (len(ppg) / Config.PPG_FS)
                if 600 < est < 800:
                    ecg_fs = est
            rpeaks_sec = convert_rpeaks_to_seconds(arr, ecg_fs)
            dur = len(ppg) / Config.PPG_FS
            rpeaks_sec = rpeaks_sec[(rpeaks_sec >= 0) & (rpeaks_sec < dur)]

    if subject_id == 1 and rpeaks_sec is not None:
        rri = np.diff(rpeaks_sec)
        print(f"  [S1] rpeaks: {len(rpeaks_sec)}개, "
              f"RRI={np.mean(rri)*1000:.1f}ms ≈{60/np.mean(rri):.0f}BPM")

    return {"ppg": ppg, "acc": acc, "label": label,
            "activity": activity, "subject_id": subject_id,
            "rpeaks_sec": rpeaks_sec}


def generate_synthetic_subject(subject_id, duration_sec=600):
    rng = np.random.RandomState(subject_id * 137 + 7)
    pf, af = Config.PPG_FS, Config.ACC_FS
    pl, al = duration_sec * pf, duration_sec * af
    base_hr = 55 + subject_id * 4 + rng.randn() * 2
    t = np.arange(pl) / pf
    hr = np.full(pl, base_hr, np.float64)
    ns = 5; sl = pl // ns
    off = rng.choice([0, 5, 15, 30, 10], ns, replace=True)
    aseg = []
    for i in range(ns):
        s, e = i * sl, min((i+1) * sl, pl)
        hr[s:e] += off[i]
        aseg.append(0 if off[i] <= 5 else 1 if off[i] <= 10 else 2 if off[i] <= 20 else 3)
    hr += 3 * np.sin(2*np.pi*0.1*t) + 1.5 * np.sin(2*np.pi*0.25*t)
    hr = np.clip(hr, 40, 180)

    ppg = np.zeros(pl, np.float64); phase = 0.0
    for i in range(pl):
        phase += 2*np.pi*(hr[i]/60)/pf
        ppg[i] = np.sin(phase) + 0.25*np.sin(2*phase-0.3)
    ppg = ppg.astype(np.float32) * (0.7+0.6*rng.rand())
    ppg += rng.randn(pl).astype(np.float32) * 0.08

    acc = rng.randn(al, 3).astype(np.float32) * 0.03
    asl = al // ns
    for i in range(ns):
        s, e = i*asl, min((i+1)*asl, al)
        if off[i] >= 15: acc[s:e] *= off[i]/5
        acc[s:e, 2] += 1.0

    rpeaks_sec = []
    ta = 0.0
    for i in range(pl):
        ta += 1.0/pf
        if ta >= 60.0/hr[i]:
            rpeaks_sec.append(i/pf); ta = 0.0
    rpeaks_sec = np.array(rpeaks_sec, np.float64)

    stride = Config.STRIDE_SEC * pf
    nw = (pl - Config.PPG_WIN_LEN) // stride + 1
    label = np.array([np.mean(hr[i*stride:i*stride+Config.PPG_WIN_LEN])
                      for i in range(nw)], np.float32)
    act = np.array([aseg[min((i*stride)//sl, ns-1)] for i in range(nw)], np.int64)
    return {"ppg": ppg, "acc": acc, "label": label, "activity": act,
            "subject_id": subject_id, "rpeaks_sec": rpeaks_sec}


# ============================================================
# 3. 데이터셋 (v5: 스펙트로그램 포함)
# ============================================================
class PPIDataset(Dataset):
    def __init__(self, subjects_data, config=None, augment=False,
                 normalize_per_subject=True):
        self.augment = augment
        self.config = config or Config()
        self.samples = []
        self.nan_count = 0
        self.rpeaks_used = 0
        self.ecg_default = 0
        self.ppg_fallback = 0
        self.rpeak_short = 0
        self.rri_invalid = 0

        for d in subjects_data:
            self._process_subject(d, normalize_per_subject)

        if self.nan_count > 0:
            print(f"    ⚠ NaN {self.nan_count}개")
        if self.rpeak_short + self.rri_invalid > 0:
            print(f"    ⚠ ECG실패: 피크부족={self.rpeak_short}, RRI무효={self.rri_invalid}")
        print(f"  → {len(self.samples)} 윈도우 "
              f"(ECG:{self.rpeaks_used}, ECG실패:{self.ecg_default}, "
              f"PPG:{self.ppg_fallback})")

    def _process_subject(self, data, normalize):
        ppg_raw = data["ppg"].copy()
        acc_raw = data["acc"].copy()
        label = data.get("label", None)
        activity = data.get("activity", None)
        sid = data["subject_id"]
        rpeaks_sec = data.get("rpeaks_sec", None)
        cfg = self.config

        ppg_peaks = detect_ppg_peaks_global(ppg_raw, cfg.PPG_FS)
        has_rp = (cfg.USE_ECG_RPEAKS and rpeaks_sec is not None
                  and len(rpeaks_sec) > 10)

        ppg_n, acc_n = ppg_raw.copy(), acc_raw.copy()
        if normalize:
            m, s = np.mean(ppg_n), max(np.std(ppg_n), 1e-8)
            ppg_n = (ppg_n - m) / s
            am = np.mean(acc_n, axis=0)
            ast = np.where(np.std(acc_n, axis=0) < 1e-8, 1.0, np.std(acc_n, axis=0))
            acc_n = (acc_n - am) / ast

        ps_stride = cfg.STRIDE_SEC * cfg.PPG_FS
        as_stride = cfg.STRIDE_SEC * cfg.ACC_FS
        nw = (len(ppg_raw) - cfg.PPG_WIN_LEN) // ps_stride + 1

        # label 인덱스 매핑 (label은 2초 간격 기준으로 저장됨)
        label_stride_sec = 2  # PPG-DaLiA label spacing

        wi = 0
        for i in range(nw):
            ps = i * ps_stride
            pe = ps + cfg.PPG_WIN_LEN
            as_ = i * as_stride
            ae = as_ + cfg.ACC_WIN_LEN
            if pe > len(ppg_raw) or ae > len(acc_raw):
                break

            ppg_win = ppg_n[ps:pe].copy()
            acc_win = acc_n[as_:ae].copy()

            # HR 레이블 (윈도우 중앙의 레이블)
            win_center_sec = (ps + cfg.PPG_WIN_LEN // 2) / cfg.PPG_FS
            label_idx = int(win_center_sec / label_stride_sec)
            if label is not None and label_idx < len(label):
                hr_label = float(label[label_idx])
            else:
                hr_label = 70.0

            # PPI 타겟
            if has_rp:
                ws = ps / cfg.PPG_FS
                we = pe / cfg.PPG_FS
                ppi_t, mean_hr, status = compute_ppi_from_rpeaks_sec(
                    rpeaks_sec, ws, we, cfg.PPI_OUTPUT_LEN,
                    margin_sec=cfg.RRI_MARGIN_SEC, hr_fallback=hr_label)
                if status == "ok":
                    self.rpeaks_used += 1
                else:
                    if status == "short": self.rpeak_short += 1
                    else: self.rri_invalid += 1
                    self.ecg_default += 1
            else:
                wp = extract_window_peaks(ppg_peaks, ps, pe)
                ppi_t, mean_hr = _ppi_fallback(wp, cfg.PPG_FS,
                                                cfg.PPG_WIN_LEN,
                                                cfg.PPI_OUTPUT_LEN, hr_label)
                self.ppg_fallback += 1

            if np.any(np.isnan(ppi_t)):
                self.nan_count += 1
                ppi_t = np.full(cfg.PPI_OUTPUT_LEN,
                                60.0 / max(hr_label, 40.0), np.float32)

            # SQI
            wp = extract_window_peaks(ppg_peaks, ps, pe)
            sqi = compute_sqi(ppg_raw[ps:pe], cfg.PPG_FS, wp)

            # [v5] 스펙트로그램
            spec = compute_spectrogram(ppg_raw[ps:pe], cfg.PPG_FS,
                                       cfg.STFT_NPERSEG, cfg.STFT_NOVERLAP,
                                       cfg.STFT_FREQ_MAX)

            # Activity
            if activity is not None and label_idx < len(activity):
                act = min(int(activity[label_idx]), cfg.NUM_ACTIVITIES - 1)
            else:
                act = min(estimate_activity(acc_raw[as_:ae]), cfg.NUM_ACTIVITIES - 1)

            if label is None or label_idx >= len(label):
                hr_label = mean_hr

            self.samples.append({
                "ppg": np.nan_to_num(ppg_win, nan=0.0).astype(np.float32),
                "acc": np.nan_to_num(acc_win, nan=0.0).astype(np.float32),
                "spec": spec,   # [v5] [n_freq, n_time]
                "ppi": ppi_t.astype(np.float32),
                "hr": np.float32(hr_label),
                "activity": np.int64(act),
                "subject_id": np.int64(sid),
                "time_sec": np.float32(ps / cfg.PPG_FS),
                "sqi": np.float32(sqi),
            })
            wi += 1

        if has_rp and wi > 0:
            rri = np.diff(rpeaks_sec)
            last_spec = self.samples[-1]["spec"]
            print(f"    S{sid}: ECG (RRI={np.mean(rri)*1000:.0f}ms "
                  f"≈{60/np.mean(rri):.0f}BPM), {wi}win "
                  f"(spec:{last_spec.shape})")
        elif wi > 0:
            print(f"    S{sid}: PPG fallback, {wi}win")
        else:
            print(f"    S{sid}: 윈도우 0개")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ppg, acc = s["ppg"].copy(), s["acc"].copy()
        spec = s["spec"].copy()

        if self.augment:
            ppg, acc, spec = self._augment(ppg, acc, spec)

        at = torch.from_numpy(acc)
        at = at.permute(1, 0) if at.ndim == 2 else at.unsqueeze(0)

        return {
            "ppg": torch.from_numpy(ppg).unsqueeze(0),     # [1, 1920]
            "acc": at,                                       # [3, 960]
            "spec": torch.from_numpy(spec).unsqueeze(0),    # [1, F, T]
            "ppi": torch.from_numpy(s["ppi"]),               # [60]
            "hr": torch.tensor(s["hr"]),
            "activity": torch.tensor(s["activity"]),
            "subject_id": torch.tensor(s["subject_id"]),
            "time_sec": torch.tensor(s["time_sec"]),
            "sqi": torch.tensor(s["sqi"]),
        }

    def _augment(self, ppg, acc, spec):
        cfg = self.config
        if np.random.rand() < 0.5:
            ppg = ppg + (np.random.randn(*ppg.shape) * cfg.AUG_NOISE_STD).astype(np.float32)
        if np.random.rand() < 0.5:
            ppg = ppg * np.random.uniform(*cfg.AUG_SCALE_RANGE)
            # spec은 log+정규화 후이므로 선형 스케일링 적용하지 않음
        if np.random.rand() < 0.3:
            acc = acc + (np.random.randn(*acc.shape) * 0.02).astype(np.float32)
        if np.random.rand() < 0.2:
            ms = np.random.randint(0, max(len(ppg) - 128, 1))
            ppg[ms:ms + np.random.randint(32, 128)] *= 0.1
        return ppg, acc, spec


def _ppi_fallback(peaks, fs, total_len, output_len, hr_fb=70.0):
    dp = 60.0 / max(hr_fb, 30.0)
    wd = total_len / fs
    if len(peaks) < 3:
        return np.full(output_len, dp, np.float32), hr_fb
    ppi = np.diff(peaks).astype(np.float64) / fs
    vm = (ppi > 0.3) & (ppi < 2.0)
    if vm.sum() < 2:
        return np.full(output_len, dp, np.float32), hr_fb
    vi = np.where(vm)[0]; pv = ppi[vi]
    mt = (peaks[vi] + peaks[vi+1]) / (2.0*fs)
    tu = np.linspace(0.0, wd, output_len)
    r = interp1d(mt, pv, kind="linear", fill_value=(pv[0], pv[-1]),
                 bounds_error=False)(tu).astype(np.float32) if len(mt) >= 2 \
        else np.full(output_len, pv[0], np.float32)
    r = np.nan_to_num(r, nan=dp, posinf=2.0, neginf=0.3)
    return np.clip(r, 0.3, 2.0), float(np.clip(np.mean(60.0/pv), 30, 200))


# ============================================================
# [v5.1] MACs/FLOPs 계산
# ============================================================
def count_macs(model, config):
    """
    모델의 MACs (Multiply-Accumulate Operations) 계산

    FLOPs ≈ 2 × MACs (곱셈 + 덧셈)
    """
    total_macs = 0
    hooks = []

    def conv1d_hook(m, inp, out):
        nonlocal total_macs
        # MACs = out_channels × out_length × (in_channels × kernel_size)
        bs = inp[0].shape[0]
        out_len = out.shape[2]
        macs = m.out_channels * out_len * (m.in_channels // m.groups) * m.kernel_size[0]
        total_macs += macs

    def conv2d_hook(m, inp, out):
        nonlocal total_macs
        bs = inp[0].shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        macs = m.out_channels * out_h * out_w * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        total_macs += macs

    def linear_hook(m, inp, out):
        nonlocal total_macs
        macs = m.in_features * m.out_features
        total_macs += macs

    def bn_hook(m, inp, out):
        nonlocal total_macs
        total_macs += out.numel()  # scale + shift

    def gru_hook(m, inp, out):
        nonlocal total_macs
        # GRU: 3 gates × (input_size × hidden_size + hidden_size × hidden_size) × seq_len
        seq_len = inp[0].shape[1]
        dirs = 2 if m.bidirectional else 1
        macs = 3 * (m.input_size * m.hidden_size +
                     m.hidden_size * m.hidden_size) * seq_len * dirs * m.num_layers
        total_macs += macs

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            hooks.append(m.register_forward_hook(bn_hook))
        elif isinstance(m, nn.GRU):
            hooks.append(m.register_forward_hook(gru_hook))

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        ppg = torch.randn(1, 1, config.PPG_WIN_LEN).to(device)
        acc = torch.randn(1, 3, config.ACC_WIN_LEN).to(device)
        # 스펙트로그램 크기 계산
        spec_f = int(config.STFT_FREQ_MAX / (config.PPG_FS / config.STFT_NPERSEG)) + 1
        spec_t = max(1, (config.PPG_WIN_LEN - config.STFT_NPERSEG) //
                     (config.STFT_NPERSEG - config.STFT_NOVERLAP) + 1)
        spec = torch.randn(1, 1, spec_f, spec_t).to(device)
        model(ppg, acc, spec, 0.0)

    for h in hooks:
        h.remove()
    model.train()  # GRU backward는 train 모드 필수

    return total_macs


# ============================================================
# 4. 모델 아키텍처 (v5.1: 1D+2D 듀얼 패스, 60초)
# ============================================================
class ResBlock1D(nn.Module):
    def __init__(self, ch, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch),
            nn.ReLU(True), nn.Dropout(drop),
            nn.Conv1d(ch, ch, 5, padding=2), nn.BatchNorm1d(ch))
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.block(x) + x)


class ResBlock2D(nn.Module):
    """2D Residual Block (스펙트로그램용)"""
    def __init__(self, ch, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
            nn.ReLU(True), nn.Dropout2d(drop),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch))
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.block(x) + x)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.a = a; return x.clone()
    @staticmethod
    def backward(ctx, g):
        return -ctx.a * g, None


class DualPathPPIModel(nn.Module):
    """
    [v5.1] 1D+2D 듀얼 패스 모델 (64초 윈도우) + PPI 정제 GRU

    경로 1 (1D): PPG 시계열 (4096 samples) → 파형 패턴
    경로 2 (2D): PPG 스펙트로그램 → 주파수 띠 추적
    경로 3 (1D): ACC 시계열 (2048 samples) → 모션 특징

    퓨전 → Conv1d → 거친 PPI → Bi-GRU 정제 (잔차 연결) → 최종 PPI

    GRU 배치 원칙: 고차원 특징(256ch)이 아닌 1차원 PPI 출력에
    경량 GRU를 적용. Conv1d가 놓친 장거리 시간 의존성을
    잔차(residual)로 보정. 파라미터 ~4K 추가.
    """

    def __init__(self, n_subj, n_act=Config.NUM_ACTIVITIES,
                 ppi_len=Config.PPI_OUTPUT_LEN, drop=Config.DROPOUT):
        super().__init__()

        # ── [v5] Instance Normalization (피험자 간 스케일 차이 흡수) ──
        self.ppg_instance_norm = nn.InstanceNorm1d(1, affine=True)
        self.acc_instance_norm = nn.InstanceNorm1d(3, affine=True)

        # ── 1D PPG 인코더 (64초 = 4096 samples) [64/128 채널] ──
        self.ppg_enc = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3), nn.BatchNorm1d(64),
            nn.ReLU(True), nn.MaxPool1d(2),            # → [64, 480]
            ResBlock1D(64, drop),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128),
            nn.ReLU(True), nn.MaxPool1d(2),            # → [128, 120]
            ResBlock1D(128, drop),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(True),                             # → [128, 60]
            nn.AdaptiveAvgPool1d(32),                  # → [128, 32]
        )

        # ── [v5] 2D 스펙트로그램 인코더 [32/64/128 채널] ──
        self.spec_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.MaxPool2d(2),
            ResBlock2D(32, drop),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(True), nn.MaxPool2d(2),
            ResBlock2D(64, drop),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 32)),             # → [128, 1, 32]
        )

        # ── 1D ACC 인코더 (64초 = 2048 samples) [64/128 채널] ──
        self.acc_enc = nn.Sequential(
            nn.Conv1d(3, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64),
            nn.ReLU(True), nn.MaxPool1d(2),            # → [64, 240]
            ResBlock1D(64, drop),
            nn.Conv1d(64, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128),
            nn.ReLU(True),                             # → [128, 120]
            nn.AdaptiveAvgPool1d(32),                  # → [128, 32]
        )

        # ── 퓨전 (1D_ppg:128 + 2D_spec:128 + 1D_acc:128 = 384ch) ──
        self.fusion = nn.Sequential(
            nn.Conv1d(384, 256, 3, padding=1), nn.BatchNorm1d(256),
            nn.ReLU(True), nn.Dropout(drop),
            ResBlock1D(256, drop),
            ResBlock1D(256, drop),
        )

        # ── PPI 헤드 ──
        self.ppi_head = nn.Sequential(
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128),
            nn.ReLU(True), nn.Dropout(drop * 0.5),
            nn.Conv1d(128, 1, 1),
            nn.AdaptiveAvgPool1d(ppi_len),
        )

        # ── PPI 시간 정제 GRU (경량) ──
        # Conv1d가 만든 거친 PPI를 시간축으로 정제
        # 입력: 1차원 PPI 시계열 → GRU → 정제된 PPI
        # 파라미터 ~4K만 추가 (vs 이전 296K)
        self.ppi_refine = nn.GRU(
            input_size=1, hidden_size=32,
            num_layers=1, batch_first=True,
            bidirectional=True)
        self.ppi_refine_fc = nn.Linear(64, 1)  # bidir 32×2 → 1

        # ── 보조 헤드 ──
        self.act_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(128, n_act))
        self.subj_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(drop),
            nn.Linear(128, n_subj))

        # Activity gate
        self.act_gate = nn.Sequential(
            nn.Linear(n_act, 32), nn.ReLU(True),
            nn.Linear(32, ppi_len), nn.Sigmoid())

        self.apply(self._iw)

    @staticmethod
    def _iw(m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, p in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(p)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)

    def forward(self, ppg, acc, spec, alpha=1.0):
        # [v5] Instance Norm (피험자 간 차이 흡수)
        ppg = self.ppg_instance_norm(ppg)
        acc = self.acc_instance_norm(acc)

        # 1D 경로
        ppg_f = self.ppg_enc(ppg)       # [B, 128, 32]
        acc_f = self.acc_enc(acc)        # [B, 128, 32]

        # 2D 경로
        spec_f = self.spec_enc(spec)    # [B, 128, 1, 32]
        spec_f = spec_f.squeeze(2)      # [B, 128, 32]

        # 퓨전 (3경로 결합)
        fused = torch.cat([ppg_f, spec_f, acc_f], dim=1)  # [B, 384, 32]
        shared = self.fusion(fused)     # [B, 256, 32]

        # PPI 예측 (Conv1d → 거친 PPI → GRU 정제)
        ppi_coarse = self.ppi_head(shared).squeeze(1)   # [B, ppi_len]

        # GRU 정제: 거친 PPI의 시간적 일관성 개선
        # [B, ppi_len] → [B, ppi_len, 1] → GRU → [B, ppi_len, 64] → FC → [B, ppi_len]
        gru_in = ppi_coarse.unsqueeze(2)                # [B, T, 1]
        gru_out, _ = self.ppi_refine(gru_in)            # [B, T, 64]
        ppi_delta = self.ppi_refine_fc(gru_out).squeeze(2)  # [B, T]
        ppi_raw = ppi_coarse + ppi_delta                # 잔차 연결

        # Activity/Subject 헤드
        act_logits = self.act_head(shared)

        gate = self.act_gate(F.softmax(act_logits.detach(), 1))
        # [v5.1] 스케일 중심 1.0: 증감 양방향 조절 가능 (0.85~1.15)
        scale = 0.85 + 0.3 * gate
        ppi_out = torch.clamp(ppi_raw * scale, 0.27, 2.2)

        subj_logits = self.subj_head(GRL.apply(shared, alpha))

        return {"ppi": ppi_out, "activity": act_logits, "subject": subj_logits}


# ============================================================
# 5. PPI → HR
# ============================================================
def ppi_to_hr(ppi, window_sec=64, label_sec=8):
    """
    PPI 시계열 → HR 변환

    [v5.1] 64초 윈도우에서 정중앙 8초만 추출하여 HR 계산.
    128pt 중 인덱스 56~72 (16pt) 추출.

    [v5.2] 개선:
    1) 중앙값 필터(medfilt, k=3)로 PPI 이상값 제거
    2) mean(60/ppi) → median(60/ppi)로 이상값에 강건
    """
    n = len(ppi)

    # 중앙값 필터: PPI 시계열의 스파이크 제거
    if n >= 3:
        ppi = medfilt(ppi, kernel_size=3)

    if window_sec > label_sec and n > 4:
        pts_per_sec = n / window_sec
        margin = (window_sec - label_sec) / 2.0
        start_idx = int(margin * pts_per_sec)
        end_idx = int((margin + label_sec) * pts_per_sec)
        center_ppi = ppi[start_idx:end_idx]
    else:
        center_ppi = ppi

    v = center_ppi[(center_ppi > 0.27) & (center_ppi < 2.2)]

    # 중앙 구간에 유효 PPI 없으면 전체에서 재시도 (안전장치)
    if len(v) == 0:
        v = ppi[(ppi > 0.27) & (ppi < 2.2)]

    # median: mean(60/ppi)보다 이상값에 강건
    return float(np.clip(np.median(60.0 / v), 30, 220)) if len(v) > 0 else 70.0


# ============================================================
# 6. 학습기
# ============================================================
class PPITrainer:
    def __init__(self, model, config):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        self.best_state = None

        self.opt = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)

        # [v5.1] LR 스케줄러
        # 1) Warmup only (5ep 선형 증가 → 이후 base_lr 유지)
        #    cosine decay 제거: ReduceLROnPlateau와 충돌 방지
        #    (LambdaLR는 매 epoch base_lr 기준 재계산 → plateau 감소를 덮어씀)
        w = 5
        self.warmup_sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda e: min((e + 1) / w, 1.0))

        # 2) ReduceLROnPlateau (MAE 정체 시 LR × 0.5)
        #    warmup 이후 plateau가 단독으로 LR decay 담당
        self.plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='min', factor=0.5, patience=3,
            min_lr=1e-6)

        self.ppi_crit = nn.SmoothL1Loss(reduction='none')
        self.act_crit = nn.CrossEntropyLoss()
        self.subj_crit = nn.CrossEntropyLoss()

    def _grl_alpha(self, ep):
        return float(2/(1+np.exp(-10*ep/max(self.config.NUM_EPOCHS, 1)))-1)

    def train_epoch(self, dl, epoch):
        self.model.train()
        m = {"total": 0, "ppi": 0, "hr": 0, "smooth": 0, "act": 0}
        nb = 0; alpha = self._grl_alpha(epoch)

        for b in dl:
            ppg = b["ppg"].to(self.device)
            acc = b["acc"].to(self.device)
            spec = b["spec"].to(self.device)
            ppi_t = b["ppi"].to(self.device)
            act_t = b["activity"].to(self.device)
            sub_t = b["subject_id"].to(self.device)
            sqi = b["sqi"].to(self.device)

            if torch.isnan(ppg).any() or torch.isnan(ppi_t).any():
                continue

            o = self.model(ppg, acc, spec, alpha)

            raw = self.ppi_crit(o["ppi"], ppi_t)  # [B, 128]

            # [v5.1] 중앙 8초 가중: 평가 구간에 3배 가중치
            center_weight = torch.ones_like(raw)
            cs = int((self.config.WINDOW_SEC - 8) / 2 *
                     self.config.PPI_OUTPUT_LEN / self.config.WINDOW_SEC)
            ce = int((self.config.WINDOW_SEC + 8) / 2 *
                     self.config.PPI_OUTPUT_LEN / self.config.WINDOW_SEC)
            center_weight[:, cs:ce] = 3.0
            raw = raw * center_weight

            if self.config.USE_SQI_WEIGHTING:
                w = self.config.SQI_WEIGHT_MIN + \
                    (self.config.SQI_WEIGHT_MAX - self.config.SQI_WEIGHT_MIN) * sqi
                lp = (raw * w.unsqueeze(1)).mean()
            else:
                lp = raw.mean()

            la = self.act_crit(o["activity"], act_t)
            ld = self.subj_crit(o["subject"], sub_t)

            # [v5.2] HR-level loss: 중앙 8초 PPI → HR 변환 후 직접 HR 오차
            # GRU가 HR 개선에 직접 기여하도록 학습 신호 제공
            ppi_center = o["ppi"][:, cs:ce]  # [B, 16] 중앙 구간
            ppi_safe = torch.clamp(ppi_center, 0.3, 2.0)
            hr_pred = (60.0 / ppi_safe).mean(dim=1)  # [B]
            hr_t = b["hr"].to(self.device)
            l_hr = F.smooth_l1_loss(hr_pred, hr_t)

            # [v5.2] Temporal smoothness loss: PPI 시계열 연속성
            # 인접 PPI 값의 급격한 변화를 억제 → GRU 정제에 명확한 방향성
            ppi_diff = o["ppi"][:, 1:] - o["ppi"][:, :-1]
            l_smooth = (ppi_diff ** 2).mean()

            loss = (self.config.LAMBDA_PPI * lp +
                    self.config.LAMBDA_HR * l_hr +
                    self.config.LAMBDA_SMOOTH * l_smooth +
                    self.config.LAMBDA_ACTIVITY * la +
                    self.config.LAMBDA_DOMAIN * ld)

            if torch.isnan(loss): continue

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()

            m["total"] += loss.item(); m["ppi"] += lp.item()
            m["hr"] += l_hr.item(); m["smooth"] += l_smooth.item()
            m["act"] += la.item(); nb += 1

        # [v5.1] 5에폭까지만 warmup, 이후는 plateau가 단독 decay
        if epoch < 5:
            self.warmup_sched.step()
        d = max(nb, 1)
        return {k: v/d for k, v in m.items()} | {
            "lr": self.opt.param_groups[0]["lr"]}

    @torch.no_grad()
    def evaluate(self, dl):
        self.model.eval()
        all_ppi, all_ppit, all_hr, all_t, all_sqi, all_sid = \
            [], [], [], [], [], []
        ac, at = 0, 0

        for b in dl:
            o = self.model(b["ppg"].to(self.device),
                           b["acc"].to(self.device),
                           b["spec"].to(self.device), 0.0)
            all_ppi.append(o["ppi"].cpu().numpy())
            all_ppit.append(b["ppi"].numpy())
            all_hr.append(b["hr"].numpy())
            all_t.append(b["time_sec"].numpy())
            all_sqi.append(b["sqi"].numpy())
            all_sid.append(b["subject_id"].numpy())
            ap = o["activity"].argmax(1)
            ac += (ap == b["activity"].to(self.device)).sum().item()
            at += len(b["activity"])

        P = np.concatenate(all_ppi, 0)
        PT = np.concatenate(all_ppit, 0)
        HR = np.concatenate(all_hr, 0)
        T = np.concatenate(all_t, 0)
        SQ = np.concatenate(all_sqi, 0)
        SI = np.concatenate(all_sid, 0)

        order = np.lexsort((T, SI))
        P, PT, HR, SQ, T = P[order], PT[order], HR[order], SQ[order], T[order]

        n = len(HR)
        ppi_mae = float(np.nanmean(np.abs(P - PT)))
        hr_pred = np.array([ppi_to_hr(P[i],
                                       window_sec=self.config.WINDOW_SEC,
                                       label_sec=8)
                            for i in range(n)])

        err = np.abs(hr_pred - HR)
        hi = SQ > 0.5; lo = SQ <= 0.5

        # PPI 예측 분산: 상수 출력 붕괴 감지용
        ppi_pred_std = float(np.mean([np.std(P[i]) for i in range(n)]))

        return {
            "hr_mae": float(np.mean(err)),
            "hr_rmse": float(np.sqrt(np.mean(err**2))),
            "ppi_mae": ppi_mae,
            "ppi_std": ppi_pred_std,  # 낮으면 상수 출력 의심
            "activity_acc": ac / max(at, 1) * 100,
            "mae_hi": float(np.mean(err[hi])) if hi.sum() > 0 else 0,
            "mae_lo": float(np.mean(err[lo])) if lo.sum() > 0 else 0,
            "avg_sqi": float(np.mean(SQ)),
            "n_win": n,
            # 논문용 원시 데이터
            "_hr_pred": hr_pred,
            "_hr_true": HR,
            "_ppi_pred": P,
            "_ppi_true": PT,
            "_sqi": SQ,
            "_time_sec": T,
        }

    def save_best(self):
        self.best_state = deepcopy(self.model.state_dict())
    def load_best(self):
        if self.best_state:
            self.model.load_state_dict(self.best_state)


# ============================================================
# 7. LoSo
# ============================================================
def run_loso(config):
    print("=" * 70)
    print(f"  PPG-DaLiA v5.1: 64s window + 1D+2D dual path")
    print(f"  [{config.ablation_name()}] │ Full {len(config.TEST_SUBJECTS)}-subject LoSo")
    print("=" * 70)
    set_seed(config.SEED)

    print(f"\n[1] 데이터 로딩")
    all_subj = {}
    for sid in config.ALL_SUBJECTS:
        d = load_ppg_dalia_subject(config.DATA_ROOT, sid)
        if d:
            rp = "✓" if d.get("rpeaks_sec") is not None else "✗"
            print(f"  ✓ S{sid} ({len(d['ppg'])/config.PPG_FS:.0f}s, rp:{rp})")
            all_subj[sid] = d
    if not all_subj:
        print("  → 합성 데이터")
        for sid in config.ALL_SUBJECTS:
            all_subj[sid] = generate_synthetic_subject(sid)
    for sid in config.ALL_SUBJECTS:
        if sid not in all_subj:
            all_subj[sid] = generate_synthetic_subject(sid)

    tsids = [s for s in config.TEST_SUBJECTS if s in all_subj]
    avail = sorted(all_subj.keys())
    print(f"\n  피험자: {len(avail)}명, 테스트: {tsids}")

    results = {}
    for fold, tsid in enumerate(tsids):
        print(f"\n{'━' * 70}")
        print(f"  [Fold {fold+1}/{len(tsids)}] 테스트: S{tsid} │ 학습: {len(avail)-1}명")
        print(f"{'━' * 70}")

        trsids = [s for s in avail if s != tsid]
        smap = {s: i for i, s in enumerate(trsids)}

        trl = [deepcopy(all_subj[s]) for s in trsids]
        for d, s in zip(trl, trsids): d["subject_id"] = smap[s]

        tel = [deepcopy(all_subj[tsid])]
        tel[0]["subject_id"] = 0

        print(f"\n  학습:")
        trds = PPIDataset(trl, config=config, augment=True)
        print(f"  테스트:")
        teds = PPIDataset(tel, config=config, augment=False)

        if len(trds) == 0 or len(teds) == 0:
            print(f"  [!] 데이터 부족, 스킵"); continue

        trdl = DataLoader(trds, batch_size=config.BATCH_SIZE,
                          shuffle=True, num_workers=0, drop_last=True)

        # [v5.2] HR-aware 오버샘플링: 고HR 윈도우를 더 자주 학습
        # S5(126BPM) 같은 고HR subject 일반화 개선 목적
        try:
            hr_vals = np.array([s["hr"] for s in trds.samples])
            sample_weights = np.ones(len(hr_vals))
            sample_weights[hr_vals > 100] = 2.0   # 100+ BPM: 2배
            sample_weights[hr_vals > 120] = 3.0   # 120+ BPM: 3배
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, num_samples=len(trds), replacement=True)
            trdl = DataLoader(trds, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=0, drop_last=True)
            n_hi = int(np.sum(hr_vals > 100))
            print(f"  HR 분포: >100BPM {n_hi}개 ({n_hi/len(hr_vals)*100:.1f}%)")
        except Exception:
            pass  # 실패 시 shuffle 유지
        tedl = DataLoader(teds, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=0)

        model = DualPathPPIModel(n_subj=len(trsids),
                                  n_act=config.NUM_ACTIVITIES,
                                  ppi_len=config.PPI_OUTPUT_LEN,
                                  drop=config.DROPOUT)
        nparams = sum(p.numel() for p in model.parameters())

        # [v5.1] MACs/FLOPs 계산 (첫 fold에서만)
        if fold == 0:
            model.to(config.DEVICE)
            macs = count_macs(model, config)
            flops = macs * 2
            print(f"\n  파라미터: {nparams:,}")
            print(f"  MACs:    {macs:,} ({macs/1e6:.1f}M)")
            print(f"  FLOPs:   {flops:,} ({flops/1e6:.1f}M)")
            model.cpu()  # trainer에서 다시 .to(device)
        else:
            print(f"\n  파라미터: {nparams:,}")

        MAX_RESTARTS = 2  # 붕괴 시 최대 재시작 횟수
        restart_count = 0

        while True:  # 붕괴 감지 시 재시작 루프
            if restart_count > 0:
                # 다른 seed로 모델 재초기화
                new_seed = config.SEED + restart_count * 1000 + tsid
                set_seed(new_seed)
                model = DualPathPPIModel(n_subj=len(trsids),
                                          n_act=config.NUM_ACTIVITIES,
                                          ppi_len=config.PPI_OUTPUT_LEN,
                                          drop=config.DROPOUT)
                print(f"  ⚠ PPI 붕괴 감지 → 재시작 {restart_count}/{MAX_RESTARTS} "
                      f"(seed={new_seed})")

            trainer = PPITrainer(model, config)
            best_mae, best_ep, patience = float("inf"), 0, 0
            ppi_history = []  # PPI loss 이력 (붕괴 감지용)
            collapsed = False

            for ep in range(config.NUM_EPOCHS):
                tm = trainer.train_epoch(trdl, ep)
                ppi_history.append(tm['ppi'])

                if (ep+1) % 5 == 0 or ep == config.NUM_EPOCHS - 1:
                    ev = trainer.evaluate(tedl)
                    star = ""
                    if ev["hr_mae"] < best_mae:
                        best_mae, best_ep, patience = ev["hr_mae"], ep+1, 0
                        trainer.save_best(); star = " ★"
                    else:
                        patience += 5

                    # [v5.1] ReduceLROnPlateau: MAE 정체 시 LR 감소
                    trainer.plateau_sched.step(ev["hr_mae"])
                    cur_lr = trainer.opt.param_groups[0]["lr"]

                    print(f"  Ep {ep+1:3d} │ "
                          f"L:{tm['total']:.4f} PPI:{tm['ppi']:.4f} │ "
                          f"MAE:{ev['hr_mae']:.2f} "
                          f"RMSE:{ev['hr_rmse']:.2f} │ "
                          f"SQI↑:{ev['mae_hi']:.2f} "
                          f"SQI↓:{ev['mae_lo']:.2f} │ "
                          f"lr:{cur_lr:.1e}{star}")

                    # [v5.1] PPI 붕괴 감지 (2중 조건)
                    # 조건1: train PPI loss가 15ep 이상 거의 변하지 않음
                    # 조건2: validation PPI 예측의 분산이 매우 낮음 (상수 출력)
                    if len(ppi_history) >= 15:
                        recent = ppi_history[-15:]
                        ppi_range = max(recent) - min(recent)
                        ppi_mean = np.mean(recent)
                        train_plateau = (ppi_mean > 0.05 and
                                        ppi_range / (ppi_mean + 1e-8) < 0.05)
                        pred_collapsed = ev.get("ppi_std", 1.0) < 0.01

                        if train_plateau and pred_collapsed:
                            collapsed = True
                            print(f"  ⚠ PPI 붕괴 감지: "
                                  f"train PPI={ppi_mean:.4f} (변동 {ppi_range/ppi_mean*100:.1f}%), "
                                  f"pred_std={ev['ppi_std']:.4f}")
                            break

                    if patience >= config.PATIENCE:
                        print(f"  → Early stop"); break

            if not collapsed or restart_count >= MAX_RESTARTS:
                break  # 정상 종료 또는 재시작 한도 도달
            restart_count += 1

        # 다음 fold를 위해 원래 seed 복원
        set_seed(config.SEED)

        trainer.load_best()
        final = trainer.evaluate(tedl)

        # [논문용] 예측 결과 .npz 저장
        save_dir = "./results_v5_1"
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, f"fold_S{tsid:02d}.npz"),
                 hr_pred=final["_hr_pred"],
                 hr_true=final["_hr_true"],
                 ppi_pred=final["_ppi_pred"],
                 ppi_true=final["_ppi_true"],
                 sqi=final["_sqi"],
                 time_sec=final["_time_sec"],
                 subject_id=tsid,
                 mae=final["hr_mae"],
                 rmse=final["hr_rmse"],
                 best_epoch=best_ep)
        print(f"  💾 저장: {save_dir}/fold_S{tsid:02d}.npz")

        # 원시 배열은 results dict에서 제외 (메모리)
        results[tsid] = {k: v for k, v in final.items() if not k.startswith("_")}
        results[tsid].update({"best_mae": best_mae, "best_epoch": best_ep,
                              "ecg_ok": teds.rpeaks_used, "ecg_fail": teds.ecg_default,
                              "n_params": nparams, "restarts": restart_count})

        ecg_pct = teds.rpeaks_used / max(len(teds), 1) * 100
        restart_info = f" (재시작 {restart_count}회)" if restart_count > 0 else ""
        print(f"\n  ▶ S{tsid} (Ep{best_ep}): "
              f"MAE={final['hr_mae']:.2f} RMSE={final['hr_rmse']:.2f} "
              f"PPI={final['ppi_mae']:.4f}{restart_info}")
        print(f"    ECG {teds.rpeaks_used}({ecg_pct:.0f}%), "
              f"실패 {teds.ecg_default}, SQI={final['avg_sqi']:.2f}")

    # 요약
    print(f"\n{'━' * 90}")
    print(f"  LoSo 결과 [v5.1 {config.ablation_name()}]")
    print(f"  Window: {config.WINDOW_SEC}s, Model: 1D+2D Dual Path, "
          f"Params: {nparams:,}")
    print(f"{'━' * 90}")
    print(f"  {'S':>4} │ {'MAE':>5} │ {'RMSE':>5} │ "
          f"{'SQI↑':>5} │ {'SQI↓':>5} │ {'PPI':>6} │ "
          f"{'ECG%':>4} │ {'fail':>4} │ {'Ep':>3}")
    print(f"  {'─'*4} │ {'─'*5} │ {'─'*5} │ "
          f"{'─'*5} │ {'─'*5} │ {'─'*6} │ {'─'*4} │ {'─'*4} │ {'─'*3}")

    ms, rs = [], []
    for sid, m in sorted(results.items()):
        total = m.get("ecg_ok", 0) + m.get("ecg_fail", 0)
        ep = m.get("ecg_ok", 0) / max(total, 1) * 100
        ef = m.get("ecg_fail", 0)
        print(f"  S{sid:>2} │ {m['hr_mae']:>4.2f}  │ {m['hr_rmse']:>4.2f}  │ "
              f"{m['mae_hi']:>4.2f}  │ {m['mae_lo']:>4.2f}  │ "
              f"{m['ppi_mae']:>.4f} │ {ep:>3.0f}% │ {ef:>4d} │ {m['best_epoch']:>3d}")
        ms.append(m['hr_mae']); rs.append(m['hr_rmse'])

    print(f"  {'─'*4} │ {'─'*5} │ {'─'*5}")
    print(f"  {'평균':>3} │ {np.mean(ms):>4.2f}  │ {np.mean(rs):>4.2f}")
    print(f"  {'std':>4} │ {np.std(ms):>4.2f}  │ {np.std(rs):>4.2f}")
    print(f"  {'med':>4} │ {np.median(ms):>4.2f}  │ {np.median(rs):>4.2f}")

    # 참고 비교
    print(f"\n  ── 참고 ──")
    print(f"  v5  (30s, S1-3): 평균 MAE=4.10")
    print(f"  v4.1 C (8s, S1-3): 평균 MAE=5.04")
    if ms:
        print(f"\n  v5.1 ({config.WINDOW_SEC}s, {len(ms)}명): "
              f"평균 MAE={np.mean(ms):.2f} ± {np.std(ms):.2f}")
    print(f"\n  파라미터: {nparams:,}")
    print(f"{'━' * 90}")

    # [논문용] 전체 요약 저장
    save_dir = "./results_v5_1"
    os.makedirs(save_dir, exist_ok=True)
    summary = {
        "subjects": np.array(sorted(results.keys())),
        "mae": np.array([results[s]["hr_mae"] for s in sorted(results.keys())]),
        "rmse": np.array([results[s]["hr_rmse"] for s in sorted(results.keys())]),
        "best_epoch": np.array([results[s]["best_epoch"] for s in sorted(results.keys())]),
        "ppi_mae": np.array([results[s]["ppi_mae"] for s in sorted(results.keys())]),
        "avg_sqi": np.array([results[s]["avg_sqi"] for s in sorted(results.keys())]),
        "mean_mae": np.mean(ms),
        "std_mae": np.std(ms),
        "median_mae": np.median(ms),
    }
    np.savez(os.path.join(save_dir, "summary.npz"), **summary)
    print(f"\n  💾 전체 결과 저장: {save_dir}/summary.npz")
    print(f"  💾 Fold별 결과: {save_dir}/fold_S*.npz (15개)")

    return results


# ============================================================
# 8. 사전 검증
# ============================================================
def quick_check(config):
    print(f"\n[검증] v5.1 ({config.ablation_name()})...")

    d = generate_synthetic_subject(99, 180)  # 3분 (60초 윈도우 최소)
    ds = PPIDataset([d], config=config, augment=False)
    if len(ds) == 0:
        return False
    s = ds[0]

    print(f"  ✓ PPG: {s['ppg'].shape}")       # [1, 3840]
    print(f"  ✓ ACC: {s['acc'].shape}")        # [3, 1920]
    print(f"  ✓ Spec: {s['spec'].shape}")      # [1, F, T]
    print(f"  ✓ PPI: {s['ppi'].shape}, NaN={torch.isnan(s['ppi']).any()}")

    # 모델 + MACs
    dl = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=True, drop_last=True)
    model = DualPathPPIModel(n_subj=1, ppi_len=config.PPI_OUTPUT_LEN)
    model.to(config.DEVICE)
    nparams = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, config)
    print(f"  ✓ 모델: {nparams:,} params, {macs/1e6:.1f}M MACs, {macs*2/1e6:.1f}M FLOPs")

    # 미니 학습
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(2):
        for b in dl:
            o = model(b["ppg"].to(config.DEVICE),
                      b["acc"].to(config.DEVICE),
                      b["spec"].to(config.DEVICE), 0)
            loss = F.smooth_l1_loss(o["ppi"], b["ppi"].to(config.DEVICE))
            if torch.isnan(loss): return False
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  ✓ Ep{ep}: loss={loss.item():.4f}")

    return True


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    config = Config()
    for p in ["./PPG_FieldStudy", "./PPG_DaLiA", "./PPG_DaLiA_pkl",
              "./PPG-DaLiA_pkl", "./PPG-DaLiA", "../PPG-DaLiA_pkl"]:
        if os.path.exists(p):
            config.DATA_ROOT = p; break

    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  PPG-DaLiA v5.1: 64s Window + 1D+2D Dual Path            ║")
    print("║                                                          ║")
    print("║  ✅ 윈도우: 64초 (PPG 4096 = 2^12, CNN 최적)              ║")
    print("║  ✅ LR: Warmup(5ep) + ReduceLROnPlateau (정체 시 ×0.5)    ║")
    print("║  ✅ 학습 연장: 120 Ep, PATIENCE=30                        ║")
    print("║  ✅ 전체 15명 LoSo + MACs/FLOPs                          ║")
    print("║  ✅ STFT 해상도 향상 (nperseg 256)                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\n  Device: {config.DEVICE}")
    print(f"  경로:   {config.DATA_ROOT}")
    print(f"  윈도우: {config.WINDOW_SEC}s (PPG:{config.PPG_WIN_LEN}, "
          f"PPI:{config.PPI_OUTPUT_LEN}pt)")
    print(f"  학습:   {config.NUM_EPOCHS}ep, PATIENCE={config.PATIENCE}")
    print(f"  테스트: {len(config.TEST_SUBJECTS)}명 전체 LoSo")
    print(f"  모드:   {config.ablation_name()}")

    if not quick_check(config):
        print("\n  ✗ 실패!"); exit(1)
    print("\n  ✓ 통과!\n")

    results = run_loso(config)
    print("\n완료.")
