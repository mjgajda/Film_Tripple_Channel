from film_read_in_code.main import (
    load_or_run_clustering,
    Load_Image_and_Run_Clustering
)
from Constants import film_labels


def main():

    # Constants.py içindeki film_labels yapısını kontrol et
    multi_labels = isinstance(film_labels[0], list)

    print("=== PRE-SCAN ONLY CHECK ===")
    print("Lütfen PRE TIFF görüntüsünü seçin.")

    # Sadece PRE görüntüsünü aç ve clustering yap
    path_pre, parent_dir = load_or_run_clustering(
        "pre",
        Load_Image_and_Run_Clustering,
        film_labels,
        multiLabels=multi_labels
    )

    print("\n=== PRE-SCAN TAMAMLANDI ===")
    print("Pre tiles folder:")
    print(path_pre)

    print("\nOriginal image folder:")
    print(parent_dir)


if __name__ == "__main__":
    main()