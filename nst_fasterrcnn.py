from pathlib import Path

from manga_detection.training.runner import main


if __name__ == "__main__":
    main(
        default_model="faster_rcnn",
        default_epochs=10,
        default_batch_size=8,
        default_lr=0.01,
        default_output_name="nst-faster-rcnn",
        default_alternate_image_root=Path("./Manga109/duplicate_images"),
    )
