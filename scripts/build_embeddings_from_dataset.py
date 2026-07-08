import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.registration import RegistrationManager


def main():
    manager = RegistrationManager()
    manager.register_all_from_dataset()
    print("Done. Embeddings saved to embeddings/embeddings.pkl")


if __name__ == "__main__":
    main()