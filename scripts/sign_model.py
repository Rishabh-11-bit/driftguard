"""Sign trusted model files so DriftGuardClassifier.load() will accept them.

Usage:  python -m scripts.sign_model models/classifier.pkl [more.pkl ...]
Set DRIFTGUARD_MODEL_KEY first to use an HMAC instead of a plain SHA-256.
Only sign files you created yourself - a pickle can execute code when loaded.
"""

import sys

from driftguard.security import sign_file

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print("signed", sign_file(p))
