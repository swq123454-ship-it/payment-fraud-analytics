"""Reproducible transaction data.

Two data paths share one schema:

``synthetic`` (default)
    A generator that reproduces the statistical properties that make payment
    fraud hard. Fraud is not sprinkled independently over transactions: it
    arrives in **account-takeover episodes** - a burst of transactions on one
    card over a few hours, starting with a tiny card-testing amount and
    escalating. Episodes concentrate on the cards that transact most, victims are
    re-targeted, and a small pool of shared devices serves many cards. Those four
    facts are exactly what makes velocity, entity-count and device-sharing
    features predictive rather than decorative.

``real``
    The IEEE-CIS Fraud Detection extract (``train_transaction.csv`` and
    ``train_identity.csv`` in ``data/raw/``), normalised onto the same schema so
    every downstream module is source-agnostic.

The generator is not decoration. Kaggle downloads need credentials, so a CI job
cannot reproduce them; shipping a documented, seeded generator is what makes the
reported metrics reproducible by anyone who clones the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

# ---------------------------------------------------------------------------
# Categorical vocabularies
# ---------------------------------------------------------------------------
PRODUCT_CODES = np.array(["W", "C", "R", "H", "S"])
PRODUCT_WEIGHTS = np.array([0.585, 0.212, 0.111, 0.055, 0.037])
#: Episodes gravitate to the product codes a fraudster can monetise quickly.
EPISODE_PRODUCT_WEIGHTS = np.array([0.26, 0.34, 0.14, 0.11, 0.15])

CARD_NETWORKS = np.array(["visa", "mastercard", "discover", "amex"])
CARD_NETWORK_WEIGHTS = np.array([0.62, 0.29, 0.06, 0.03])
CARD_TYPES = np.array(["debit", "credit", "debit or credit", "charge card"])
CARD_TYPE_WEIGHTS = np.array([0.72, 0.19, 0.07, 0.02])

#: Consumer mail providers, plus the throwaway and privacy-relay domains that
#: carry disproportionate fraud risk in production data.
EMAIL_DOMAINS = np.array(
    [
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "aol.com",
        "comcast.net",
        "icloud.com",
        "live.com",
        "msn.com",
        "verizon.net",
        "att.net",
        "sbcglobal.net",
        "bellsouth.net",
        "me.com",
        "mac.com",
        "charter.net",
        "cox.net",
        "earthlink.net",
        "juno.com",
        "aim.com",
        "mail.com",
        "gmx.com",
        "yandex.ru",
        "protonmail.com",
        "anonymous.com",
        "mailinator.com",
        "guerrillamail.com",
        "tempmail.com",
    ]
)

DEVICE_INFO = np.array(
    [
        "Windows",
        "MacOS",
        "iOS Device",
        "Samsung SM-G950F",
        "Samsung SM-J700M",
        "Redmi Note 4",
        "Moto G (5) Plus",
        "LG-H870",
        "HUAWEI VNS-L31",
        "trident/7.0",
        "rv:11.0",
        "iPhone",
        "iPad",
        "BLU STUDIO X",
        "ASUS_Z00AD",
        "LENOVO A6020a40",
    ]
)

BROWSERS = np.array(
    ["chrome", "safari", "ie", "firefox", "mobile safari", "edge", "samsung browser", "opera", "android"]
)
BROWSER_WEIGHTS = np.array([0.42, 0.16, 0.05, 0.09, 0.16, 0.05, 0.03, 0.02, 0.02])

SCREENS = np.array(["1280x720", "1920x1080", "1366x768", "375x667", "360x640", "414x896", "768x1024"])
OS_NAMES = np.array(["windows 10", "windows 7", "ios 11", "android 7", "mac os x", "android 8"])

#: Devices shared across many cards - the carding-ring signature that a
#: device-to-card fan-out feature is designed to catch.
N_RING_DEVICES = 60

#: Fraction of episode transactions within which the scatter component is
#: negligible; kept as a guard so the two components can never over-shoot.
SCATTERED_FRAUD_SHARE = 0.09


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class RawExtract:
    """The two source tables plus the manifest of what generated them."""

    transactions: pd.DataFrame
    identity: pd.DataFrame
    manifest: dict = field(default_factory=dict)

    @property
    def fraud_rate(self) -> float:
        return float(self.transactions[cfg.TARGET_COLUMN].mean())

    def __len__(self) -> int:
        return len(self.transactions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_timestamps(rng: np.random.Generator, n: int, days: int) -> np.ndarray:
    """Seconds since epoch, with weekday and hour-of-day seasonality.

    Real transaction streams are not uniform: Friday and Saturday evenings carry
    far more volume than Tuesday at 04:00, and repeating that here keeps the
    rolling-window SQL features meaningful.
    """
    day_weights = np.array([0.135, 0.132, 0.137, 0.144, 0.163, 0.166, 0.123])
    day_weights = day_weights / day_weights.sum()
    dow = rng.choice(7, size=n, p=day_weights)
    occurrences = np.array([len(range(d, days, 7)) for d in range(7)])
    day = dow + 7 * rng.integers(0, np.maximum(occurrences[dow], 1))

    hour_weights = np.array(
        [
            0.9,
            0.6,
            0.4,
            0.3,
            0.3,
            0.5,
            1.1,
            2.0,
            3.1,
            3.8,
            4.2,
            4.4,
            4.6,
            4.7,
            4.8,
            4.9,
            5.0,
            5.1,
            5.2,
            4.9,
            4.4,
            3.6,
            2.6,
            1.7,
        ]
    )
    hour = rng.choice(24, size=n, p=hour_weights / hour_weights.sum())
    second = rng.integers(0, 3600, size=n) + rng.integers(0, 60, size=n) * 60
    offsets = day.astype(np.int64) * 86_400 + hour.astype(np.int64) * 3_600 + second
    return offsets + cfg.SIMULATION_START_EPOCH


def _popularity_weights(rng: np.random.Generator, size: int, exponent: float) -> np.ndarray:
    """Zipf-like activity weights: a few cards transact constantly, most rarely."""
    raw = rng.pareto(exponent, size) + 1.0
    return raw / raw.sum()


def _round_amounts(rng: np.random.Generator, amounts: np.ndarray) -> np.ndarray:
    """A slice of real traffic is denominated in round numbers (top-ups, gift cards)."""
    mask = rng.random(amounts.shape[0]) < 0.06
    grid = rng.choice([5.0, 10.0, 20.0, 50.0, 100.0, 200.0], size=int(mask.sum()))
    amounts[mask] = np.round(amounts[mask] / grid) * grid
    return amounts


@dataclass
class _Population:
    """Entity pools shared by the normal stream and the fraud episodes."""

    n_cards: int
    n_addrs: int
    n_devices: int
    card_weights: np.ndarray
    addr_weights: np.ndarray
    device_weights: np.ndarray
    card_risk: np.ndarray
    addr_risk: np.ndarray
    device_risk: np.ndarray


def _build_population(rng: np.random.Generator, n_transactions: int) -> _Population:
    n_cards = max(2_000, n_transactions // 30)
    n_addrs = max(400, n_transactions // 400)
    n_devices = max(800, n_transactions // 200)
    return _Population(
        n_cards=n_cards,
        n_addrs=n_addrs,
        n_devices=n_devices,
        card_weights=_popularity_weights(rng, n_cards, 1.35),
        addr_weights=_popularity_weights(rng, n_addrs, 1.60),
        device_weights=_popularity_weights(rng, n_devices, 1.20),
        card_risk=rng.normal(0.0, 1.0, n_cards),
        addr_risk=rng.normal(0.0, 1.0, n_addrs),
        device_risk=rng.normal(0.0, 1.0, n_devices),
    )


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------
def generate_synthetic_extract(
    n_transactions: int = cfg.DEFAULT_N_TRANSACTIONS,
    fraud_rate: float = cfg.DEFAULT_FRAUD_RATE,
    days: int = cfg.SIMULATION_DAYS,
    seed: int = cfg.SEED,
) -> RawExtract:
    """Generate a two-table extract with a learnable but non-trivial fraud signal.

    The fraud process is built from three components:

    1. **Account-takeover episodes.** A compromised card produces a short burst
       of transactions over hours: a small test, then escalating amounts. This is
       what gives the velocity windows something real to detect.
    2. **Repeat victimisation and shared devices.** Episode cards are drawn from
       the active pool with a strong boost for cards already attacked, and most
       episodes route through a small ring of shared devices. Those two facts are
       what make the resolution-lagged label rate and the device fan-out feature
       predictive.
    3. **Scattered fraud.** A small independent component that no feature
       explains. Without it a model would appear far better than it can be.

    Ordinary life also bursts: a small share of legitimate traffic is a shopping
    spree on one card within an afternoon, so velocity is informative rather than
    deterministic.
    """
    rng = np.random.default_rng(seed)
    population = _build_population(rng, n_transactions)

    n_episode_txns = max(int(round(n_transactions * cfg.EPISODE_VOLUME_SHARE)), 40)
    n_normal = n_transactions - n_episode_txns
    n_fraud_target = int(round(n_transactions * fraud_rate))

    episode_frame = _generate_episodes(rng, population, n_episode_txns, days, fraud_rate, n_transactions)
    normal_frame = _generate_normal_stream(rng, population, n_normal, days)

    frame = pd.concat([normal_frame, episode_frame], ignore_index=True)
    frame = frame.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    frame["TransactionID"] = np.arange(1, len(frame) + 1)

    is_episode = frame["_is_episode"].to_numpy(dtype=bool)

    # --- scattered fraud: label noise that nothing can explain ---------------
    n_episode_fraud = int(frame.loc[is_episode, cfg.TARGET_COLUMN].sum())
    n_scattered = max(n_fraud_target - n_episode_fraud, 0)
    normal_index = np.flatnonzero(~is_episode)
    if n_scattered > 0 and len(normal_index) > 0:
        # Weighted slightly toward higher-risk cards, so entity risk is a weak
        # but recoverable signal rather than pure noise.
        risk = population.card_risk[frame.loc[normal_index, "card1"].to_numpy() - 1_000]
        weights = np.exp(0.45 * risk)
        weights = weights / weights.sum()
        chosen = rng.choice(len(normal_index), size=min(n_scattered, len(normal_index)), replace=False, p=weights)
        frame.loc[normal_index[chosen], cfg.TARGET_COLUMN] = 1

    frame = _attach_mismatch_and_missingness(frame, rng, population)
    frame = _attach_vendor_counters(frame, rng)
    frame = _attach_categoricals(frame, rng)
    frame = _attach_post_hoc_fields(frame, rng)
    identity = _build_identity_table(frame, rng, population)

    frame = frame.drop(columns=["_is_episode", "device_idx", "product_code", "P_emaildomain_idx"])
    frame = frame[list(_TRANSACTION_ORDER)]

    manifest = {
        "seed": seed,
        "n_transactions": int(len(frame)),
        "target_fraud_rate": float(fraud_rate),
        "realised_fraud_rate": float(frame[cfg.TARGET_COLUMN].mean()),
        "episode_volume_share": cfg.EPISODE_VOLUME_SHARE,
        "episodes": int(episode_frame.attrs.get("n_episodes", 0)),
        "episode_transactions": int(n_episode_txns),
        "episode_fraud_transactions": int(n_episode_fraud),
        "scattered_fraud_transactions": int(n_scattered),
        "legitimate_spree_transactions": int(normal_frame.attrs.get("n_spree", 0)),
        "n_cards": int(population.n_cards),
        "ring_devices": N_RING_DEVICES,
        "identity_coverage": float(len(identity) / max(len(frame), 1)),
        "mechanisms": [
            "account-takeover episodes: short bursts on one card with escalating amounts",
            "repeat victimisation: previously attacked cards are re-targeted",
            "shared device ring: a small device pool serves many episode cards",
            "card testing: sub-3-unit opening transactions",
            "night-time preference during episodes",
            "elevated cardholder-verification mismatches during episodes",
            "legitimate sprees so that velocity is not a perfect predictor",
            "scattered fraud with no explanatory features",
        ],
    }
    return RawExtract(transactions=frame, identity=identity, manifest=manifest)


_TRANSACTION_ORDER = (
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    cfg.TARGET_COLUMN,
    "case_closed",
    "chargeback_amount",
    "chargeback_days",
)


# ---------------------------------------------------------------------------
# Component generators
# ---------------------------------------------------------------------------
def _generate_episodes(
    rng: np.random.Generator,
    population: _Population,
    n_episode_txns: int,
    days: int,
    fraud_rate: float,
    n_transactions: int,
) -> pd.DataFrame:
    """Build account-takeover episodes, processed in chronological order.

    Ordering matters: a card can only be a *repeat* victim if it was attacked
    earlier, which is what makes the 60-day lagged label rate a legitimate and
    genuinely useful feature rather than a coincidence.
    """
    fraud_within_episode = float(
        np.clip(
            fraud_rate * (1.0 - SCATTERED_FRAUD_SHARE) / cfg.EPISODE_VOLUME_SHARE,
            0.05,
            0.95,
        )
    )

    lengths = rng.choice(
        [1, 2, 3, 4, 5, 6, 8],
        size=6 * n_episode_txns,
        p=[0.30, 0.24, 0.18, 0.12, 0.08, 0.05, 0.03],
    )
    cumulative = np.cumsum(lengths)
    n_episodes = int(np.searchsorted(cumulative, n_episode_txns) + 1)
    lengths = lengths[:n_episodes]

    start_seconds = rng.integers(0, max(days * 86_400 - 3_600, 3_600), size=n_episodes)
    order = np.argsort(start_seconds, kind="stable")
    lengths = lengths[order]
    start_seconds = start_seconds[order]

    # Adversaries get better over the period. The pressure multiplier is
    # normalised to a mean of one so that the overall fraud rate still lands on
    # its target while the *shape* drifts upward through the calendar.
    progress = start_seconds / max(days * 86_400, 1)
    pressure = 1.0 + cfg.FRAUD_PRESSURE_DRIFT * progress
    pressure = pressure / pressure.mean()

    # The device ring rotates: the pool the first half of the period uses is
    # abandoned for a fresh one. This is what puts movement into the device
    # features, and it is why a model can drift without any single feature
    # breaking.
    ring_first = np.arange(population.n_devices - 2 * N_RING_DEVICES, population.n_devices - N_RING_DEVICES)
    ring_second = np.arange(population.n_devices - N_RING_DEVICES, population.n_devices)

    # Episodes concentrate on the cards that transact most, because those are the
    # accounts with something worth stealing.
    activity = population.card_weights / population.card_weights.max()
    target_pool = np.flatnonzero(activity > np.quantile(activity, 0.82))
    pool_base = population.card_weights[target_pool].copy()

    attacked_before: dict[int, int] = {}
    rows: list[pd.DataFrame] = []

    for start, length, episode_pressure in zip(start_seconds, lengths, pressure, strict=True):
        length = int(length)
        # Repeat victimisation: the more often a card has been attacked, the more
        # likely it is to be attacked again.
        boost = np.array([1.0 + 5.0 * attacked_before.get(int(card), 0) for card in target_pool])
        weights = pool_base * boost
        weights = weights / weights.sum()
        card = int(rng.choice(target_pool, p=weights))
        attacked_before[card] = attacked_before.get(card, 0) + 1

        span = int(rng.integers(3_600, 60 * 3_600))
        offsets = np.sort(rng.integers(0, span, size=length))
        times = cfg.SIMULATION_START_EPOCH + int(start) + offsets

        # A fraudster hits at night, when the victim is asleep and the issuer's
        # analyst team is thin.
        hour_of_day = (times % 86_400) // 3_600
        daytime = (hour_of_day >= 6) & (hour_of_day < 23)
        times = np.where(
            daytime & (rng.random(length) < 0.55),
            times - (times % 86_400) + 2 * 3_600,
            times,
        )

        # Amounts escalate: a small test, then progressively larger attempts, with
        # the whole distribution inflated slightly over the period.
        inflation = 1.0 + cfg.AMOUNT_INFLATION_DRIFT * (start / max(days * 86_400, 1))
        step = np.arange(length)
        amounts = (
            np.where(
                step == 0,
                np.exp(rng.normal(1.20, 0.55, length)),
                np.exp(rng.normal(4.00 + 0.30 * np.minimum(step, 5), 0.85, length)),
            )
            * inflation
        )

        ring = ring_first if start < days * 86_400 / 2 else ring_second
        from_ring = rng.random(length) < 0.72
        ring_devices = rng.choice(ring, size=length)
        organic_devices = rng.choice(population.n_devices, size=length, p=population.device_weights)

        rows.append(
            pd.DataFrame(
                {
                    "TransactionDT": times,
                    "TransactionAmt": np.clip(amounts, 0.50, 25_000.0).round(2),
                    "card1": card + 1_000,
                    "card2": rng.integers(100, 600, length).astype(float),
                    "card3": rng.choice([150.0, 185.0, 119.0], size=length),
                    "card4": rng.choice(CARD_NETWORKS, size=length, p=CARD_NETWORK_WEIGHTS),
                    "card5": rng.choice([102.0, 117.0, 226.0, 224.0], size=length),
                    "card6": rng.choice(CARD_TYPES, size=length, p=CARD_TYPE_WEIGHTS),
                    # The fraudster uses an address the issuer has not seen.
                    "addr1": np.where(
                        rng.random(length) < 0.35,
                        np.nan,
                        (rng.choice(population.n_addrs, size=length, p=population.addr_weights) + 100).astype(float),
                    ),
                    "addr2": np.round(rng.normal(60.0, 20.0, length)).clip(10, 99),
                    "dist1": np.round(np.exp(rng.normal(4.6, 1.1, length))),
                    "P_emaildomain_idx": rng.choice(
                        len(EMAIL_DOMAINS), size=length, p=_popularity_weights(rng, len(EMAIL_DOMAINS), 0.90)
                    ),
                    "device_idx": np.where(from_ring, ring_devices, organic_devices),
                    "product_code": rng.choice(PRODUCT_CODES, size=length, p=EPISODE_PRODUCT_WEIGHTS),
                    "mismatch_boost": 1.0,
                    cfg.TARGET_COLUMN: (
                        rng.random(length) < float(np.clip(fraud_within_episode * episode_pressure, 0.02, 0.98))
                    ).astype(int),
                    "_is_episode": True,
                }
            )
        )

    frame = pd.concat(rows, ignore_index=True)
    frame.attrs["n_episodes"] = n_episodes
    return frame


def _generate_normal_stream(
    rng: np.random.Generator,
    population: _Population,
    n: int,
    days: int,
) -> pd.DataFrame:
    """Ordinary traffic, plus a small share of legitimate shopping sprees."""
    timestamps = _sample_timestamps(rng, n, days)
    card_idx = rng.choice(population.n_cards, size=n, p=population.card_weights)
    addr_idx = rng.choice(population.n_addrs, size=n, p=population.addr_weights)
    email_idx = rng.choice(len(EMAIL_DOMAINS), size=n, p=_popularity_weights(rng, len(EMAIL_DOMAINS), 0.90))
    device_idx = rng.choice(population.n_devices, size=n, p=population.device_weights)

    amounts = _round_amounts(rng, np.exp(rng.normal(3.72, 1.18, size=n)))
    # Mild basket inflation across the period, so the amount distribution itself
    # drifts and the monitoring layer has something real to report.
    progress = np.clip((timestamps - cfg.SIMULATION_START_EPOCH) / (days * 86_400), 0.0, 1.0)
    amounts = amounts * (1.0 + cfg.AMOUNT_INFLATION_DRIFT * progress)
    amounts = np.clip(amounts, 0.50, 25_000.0).round(2)

    # A spree: one card, several transactions in an afternoon. Without this,
    # "many transactions in one hour" would be a perfect fraud detector, which it
    # is not in production. Sprees also become more common toward the end of the
    # period, mirroring a seasonal build-up.
    n_spree = int(round(n * cfg.LEGITIMATE_SPREE_SHARE))
    if n_spree > 0:
        spree_cards = rng.choice(population.n_cards, size=max(n_spree // 3, 1), p=population.card_weights)
        position_weights = 1.0 - cfg.SPREE_GROWTH_DRIFT / 2 + cfg.SPREE_GROWTH_DRIFT * progress
        position_weights = position_weights / position_weights.sum()
        spree_positions = rng.choice(n, size=n_spree, replace=False, p=position_weights)
        card_idx[spree_positions] = spree_cards[rng.integers(0, len(spree_cards), size=n_spree)]
        anchor = timestamps[spree_positions] - rng.integers(0, 3 * 3_600, size=n_spree)
        timestamps[spree_positions] = np.maximum(
            anchor + rng.integers(0, 3 * 3_600, size=n_spree), cfg.SIMULATION_START_EPOCH
        )

    frame = pd.DataFrame(
        {
            "TransactionDT": timestamps,
            "TransactionAmt": amounts,
            "card1": card_idx + 1_000,
            "card2": rng.integers(100, 600, n).astype(float),
            "card3": rng.choice([150.0, 185.0, 119.0], size=n),
            "card4": rng.choice(CARD_NETWORKS, size=n, p=CARD_NETWORK_WEIGHTS),
            "card5": rng.choice([102.0, 117.0, 226.0, 224.0], size=n),
            "card6": rng.choice(CARD_TYPES, size=n, p=CARD_TYPE_WEIGHTS),
            "addr1": (addr_idx + 100).astype(float),
            "addr2": np.round(rng.normal(60.0, 20.0, n)).clip(10, 99),
            "dist1": np.round(np.exp(rng.normal(4.0, 1.3, n))),
            "P_emaildomain_idx": email_idx,
            "device_idx": device_idx,
            "product_code": rng.choice(PRODUCT_CODES, size=n, p=PRODUCT_WEIGHTS),
            "mismatch_boost": 0.0,
            cfg.TARGET_COLUMN: np.zeros(n, dtype=int),
            "_is_episode": False,
        }
    )
    frame.attrs["n_spree"] = n_spree
    return frame


# ---------------------------------------------------------------------------
# Attribute completion
# ---------------------------------------------------------------------------
def _attach_mismatch_and_missingness(
    frame: pd.DataFrame, rng: np.random.Generator, population: _Population
) -> pd.DataFrame:
    """AVS/CVV flags and nulls, driven by the card's latent risk.

    Missingness is not random: it is a property of the merchant's integration and
    of the card. Keeping it as a feature is deliberate, so the mechanism that
    creates it has to exist in the generator too.
    """
    frame = frame.copy()
    card_risk = population.card_risk[frame["card1"].to_numpy() - 1_000]
    boost = frame["mismatch_boost"].to_numpy(dtype=float)

    base = np.array([-1.05, -0.80, -1.60, -0.55, -1.35, -0.95])
    probability = _logistic(base[None, :] + (0.50 * card_risk)[:, None] + (1.60 * boost)[:, None])
    draws = rng.random((len(frame), 6)) < probability
    for index in range(6):
        frame[f"M{index + 1}"] = np.where(draws[:, index], "T", "F")

    frame["card2"] = np.where(rng.random(len(frame)) < _logistic(-2.35 + 0.38 * card_risk), np.nan, frame["card2"])
    frame["card5"] = np.where(rng.random(len(frame)) < 0.07 + 0.03 * boost, np.nan, frame["card5"])
    frame["addr1"] = np.where(rng.random(len(frame)) < (0.09 + 0.10 * boost), np.nan, frame["addr1"])
    frame["dist1"] = np.where(rng.random(len(frame)) < (0.35 - 0.25 * boost), np.nan, frame["dist1"])
    return frame.drop(columns=["mismatch_boost"])


def _attach_categoricals(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Map integer entity indices back onto their labels."""
    frame = frame.copy()
    frame["ProductCD"] = frame["product_code"]
    frame["P_emaildomain"] = EMAIL_DOMAINS[frame["P_emaildomain_idx"].to_numpy()]
    frame["R_emaildomain"] = np.where(rng.random(len(frame)) < 0.72, None, frame["P_emaildomain"])
    return frame


def _attach_vendor_counters(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Vendor-supplied all-time counters.

    ``C1``..``C4`` are lifetime counts, ``C5`` is a *forward* seven-day count and
    ``C6`` is a deviation from an all-time mean. The last two look into the
    future, which is precisely why the pipeline keeps them in a separate mart and
    measures what they are worth.
    """
    frame = frame.copy()
    card = frame["card1"].to_numpy()
    addr = frame["addr1"].fillna(-1).to_numpy()
    email = frame["P_emaildomain_idx"].to_numpy()

    frame["C1"] = pd.Series(card).map(pd.Series(card).value_counts()).to_numpy(dtype=float)
    frame["C2"] = pd.Series(addr).map(pd.Series(addr).value_counts()).to_numpy(dtype=float)
    frame["C3"] = pd.Series(email).map(pd.Series(email).value_counts()).to_numpy(dtype=float)
    frame["C4"] = (
        pd.Series(card * 10_000 + addr).map(pd.Series(card * 10_000 + addr).value_counts()).to_numpy(dtype=float)
    )

    day = ((frame["TransactionDT"] - cfg.SIMULATION_START_EPOCH) // 86_400).to_numpy()
    amount = frame["TransactionAmt"].to_numpy()
    card_frame = pd.DataFrame({"card": card, "day": day, "amt": amount})
    frame["C6"] = np.round(amount - card_frame.groupby("card")["amt"].transform("mean").to_numpy(), 2)

    ordered = card_frame.sort_values("day")
    rolling = ordered.groupby("card")["day"].transform(lambda series: series.rolling(window=8, min_periods=1).count())
    forward = np.empty(len(frame))
    forward[ordered.index.to_numpy()] = rolling.to_numpy(dtype=float)
    frame["C5"] = forward
    return frame


def _attach_post_hoc_fields(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Post-chargeback fields, present only for cases old enough to be resolved.

    These exist because a real warehouse stores them, and they are dropped before
    modelling. A single careless join on ``chargeback_amount`` produces a model
    with an AUC of 1.00 that cannot be deployed - the exact failure mode this
    project sets out to make impossible.
    """
    frame = frame.copy()
    day = (frame["TransactionDT"] - cfg.SIMULATION_START_EPOCH) // 86_400
    days_elapsed = (int(day.max()) - day).to_numpy()
    frame["case_closed"] = (days_elapsed >= 60).astype(int)

    charged_back = (
        (frame[cfg.TARGET_COLUMN].to_numpy() == 1)
        & (frame["case_closed"].to_numpy() == 1)
        & (rng.random(len(frame)) < 0.82)
    )
    recovery = rng.uniform(0.15, 0.75, len(frame))
    frame["chargeback_amount"] = np.where(charged_back, (frame["TransactionAmt"].to_numpy() * recovery).round(2), 0.0)
    frame["chargeback_days"] = np.where(charged_back, days_elapsed - 60 + rng.integers(1, 30, len(frame)), 0)
    return frame


def _build_identity_table(frame: pd.DataFrame, rng: np.random.Generator, population: _Population) -> pd.DataFrame:
    """Identity records for a minority of traffic.

    Coverage is not independent of risk: episode transactions involve devices the
    issuer has not seen before, so they show up in the identity feed more often.
    The absence of a record is therefore informative, which is why the join is a
    LEFT JOIN and why ``identity_missing`` is a feature.
    """
    device_idx = frame["device_idx"].to_numpy()
    device_risk = population.device_risk[device_idx]
    is_episode = frame["_is_episode"].to_numpy(dtype=bool)

    present = rng.random(len(frame)) < _logistic(-1.45 + 0.55 * device_risk + 1.10 * is_episode.astype(float))
    n_present = int(present.sum())

    return pd.DataFrame(
        {
            "TransactionID": frame.loc[present, "TransactionID"].to_numpy(),
            "id_01": np.round(rng.normal(0.0, 2.5, n_present), 2),
            "id_02": np.round(np.exp(rng.normal(10.5, 1.1, n_present)), 2),
            "id_05": np.where(rng.random(n_present) < 0.55, rng.normal(100.0, 40.0, n_present), np.nan).round(2),
            "id_06": np.where(rng.random(n_present) < 0.45, rng.normal(20.0, 15.0, n_present), np.nan).round(2),
            "id_11": np.round(rng.choice([0.0, 100.0, 50.0], size=n_present), 2),
            "id_12": rng.choice(["Found", "NotFound"], size=n_present, p=[0.16, 0.84]),
            "id_15": rng.choice(["Found", "New", "Unknown"], size=n_present, p=[0.18, 0.12, 0.70]),
            "id_16": rng.choice(["Found", "NotFound"], size=n_present, p=[0.10, 0.90]),
            "id_31_browser": rng.choice(BROWSERS, size=n_present, p=BROWSER_WEIGHTS),
            "id_33_screen": rng.choice(SCREENS, size=n_present),
            "DeviceType": np.where(rng.random(n_present) < 0.69, "desktop", "mobile"),
            "DeviceInfo": DEVICE_INFO[device_idx[present] % len(DEVICE_INFO)],
            "id_30_os": OS_NAMES[rng.integers(0, len(OS_NAMES), n_present)],
        }
    )


# ---------------------------------------------------------------------------
# Real data loader
# ---------------------------------------------------------------------------
def load_ieee_cis(data_dir: Path | None = None) -> RawExtract:
    """Load the IEEE-CIS extract from ``data/raw`` onto the synthetic schema.

    Column names are normalised and the identity table is trimmed to the subset
    the pipeline consumes, so every downstream module works unchanged.
    """
    data_dir = Path(data_dir or cfg.DATA_RAW_DIR)
    transaction_path = data_dir / "train_transaction.csv"
    identity_path = data_dir / "train_identity.csv"
    if not transaction_path.exists() or not identity_path.exists():
        raise FileNotFoundError(
            "IEEE-CIS extract not found. Download train_transaction.csv and "
            f"train_identity.csv from Kaggle into {data_dir} and re-run with --data-mode real."
        )

    transactions = pd.read_csv(transaction_path, low_memory=False).rename(columns={"isFraud": "is_fraud"})
    identity = pd.read_csv(identity_path, low_memory=False)
    if "is_fraud" not in transactions.columns:
        raise ValueError("train_transaction.csv does not contain an isFraud column.")
    transactions["is_fraud"] = pd.to_numeric(transactions["is_fraud"], errors="coerce").fillna(0).astype(int)

    keep = [
        c
        for c in [
            "TransactionID",
            "TransactionDT",
            "TransactionAmt",
            "ProductCD",
            "card1",
            "card2",
            "card3",
            "card4",
            "card5",
            "card6",
            "addr1",
            "addr2",
            "dist1",
            "P_emaildomain",
            "R_emaildomain",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "M6",
            "is_fraud",
        ]
        if c in transactions.columns
    ]
    transactions = transactions[keep].sort_values("TransactionDT").reset_index(drop=True)
    for column in _TRANSACTION_ORDER:
        if column not in transactions.columns and column not in ("case_closed", "chargeback_amount", "chargeback_days"):
            transactions[column] = np.nan
    for column in ("case_closed", "chargeback_amount", "chargeback_days"):
        transactions[column] = 0

    if "id_31" in identity.columns:
        identity["id_31_browser"] = (
            identity["id_31"].astype(str).str.lower().str.extract(r"([a-z]+)")[0].fillna("unknown")
        )
    if "id_33" in identity.columns:
        identity["id_33_screen"] = identity["id_33"].astype(str)
    for column in ("DeviceType", "DeviceInfo", "id_31_browser", "id_33_screen"):
        if column not in identity.columns:
            identity[column] = "unknown"
    if "TransactionID" not in identity.columns:
        identity["TransactionID"] = []
    identity = identity[["TransactionID", "DeviceType", "DeviceInfo", "id_31_browser", "id_33_screen"]]

    manifest = {
        "source": "ieee-cis",
        "n_transactions": int(len(transactions)),
        "realised_fraud_rate": float(transactions["is_fraud"].mean()),
        "identity_coverage": float(identity["TransactionID"].nunique() / max(len(transactions), 1)),
    }
    return RawExtract(transactions=transactions, identity=identity, manifest=manifest)


def load_extract(data_mode: str = "synthetic", **kwargs) -> RawExtract:
    if data_mode == "synthetic":
        return generate_synthetic_extract(**kwargs)
    if data_mode == "real":
        return load_ieee_cis()
    raise ValueError(f"Unknown data mode: {data_mode!r}")


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------
def assign_time_split(transactions: pd.DataFrame) -> pd.Series:
    """Split by transaction order, never at random.

    A random split lets the model learn from transactions that happened *after*
    the ones it is scored on. Fraud patterns drift, so a random split reports a
    materially better number than a fraud team would ever see; the holdout here
    is strictly forward in time.
    """
    n = len(transactions)
    train_end = int(n * cfg.TRAIN_FRACTION)
    test_end = train_end + int(n * cfg.TEST_FRACTION)
    ranks = transactions["TransactionDT"].rank(method="first").astype(int).to_numpy()
    split = np.empty(n, dtype=object)
    split[ranks <= train_end] = "train"
    split[(ranks > train_end) & (ranks <= test_end)] = "test"
    split[ranks > test_end] = "oot"
    return pd.Series(split, index=transactions.index, name="split")
