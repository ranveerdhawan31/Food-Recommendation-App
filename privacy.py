"""Privacy helpers.

* Approximate location for discovery; precise location only at checkout.
* Consent gate before storing health/allergy/diet data.
* Field-level encryption (Fernet) for health data at rest. Keys come from your
  KMS (AWS KMS / Secrets Manager); never hard-code them. Retention policy is
  enforced by your storage layer (TTL jobs), not here.
"""
from __future__ import annotations

import json

from .schema import UserConstraints, UserProfile

APPROX_DECIMALS = 2  # ~1.1 km


class ConsentRequired(Exception):
    pass


def approximate_location(lat: float, lon: float, decimals: int = APPROX_DECIMALS) -> tuple[float, float]:
    return round(lat, decimals), round(lon, decimals)


def require_consent(profile: UserProfile) -> None:
    if not profile.consent_given:
        raise ConsentRequired("explicit consent is required before storing health/dietary data")


def constraints_to_json(c: UserConstraints) -> str:
    return json.dumps({
        "allergies": {a.value: int(s) for a, s in c.allergies.items()},
        "diets": sorted(d.value for d in c.diets),
        "halal": int(c.halal),
    })


class FieldEncryptor:
    def __init__(self, key: bytes) -> None:
        from cryptography.fernet import Fernet  # lazy import

        self._f = Fernet(key)

    @staticmethod
    def generate_key() -> bytes:
        from cryptography.fernet import Fernet

        return Fernet.generate_key()

    def encrypt_constraints(self, profile: UserProfile) -> bytes:
        require_consent(profile)
        return self._f.encrypt(constraints_to_json(profile.constraints).encode())

    def decrypt_constraints(self, token: bytes) -> UserConstraints:
        from .schema import Allergen, Diet, HalalLevel, Severity

        raw = json.loads(self._f.decrypt(token))
        return UserConstraints({Allergen(a): Severity(s) for a, s in raw["allergies"].items()},
                               {Diet(d) for d in raw["diets"]}, HalalLevel(raw["halal"]))
