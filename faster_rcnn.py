from manga_detection.training.runner import main


if __name__ == "__main__":
    main(
        default_model="faster_rcnn",
        default_epochs=10,
        default_batch_size=4,
        default_lr=1e-4,
        default_output_name="faster-rcnn_pretrained_augment_lr=0.0001_imgaug",
    )
