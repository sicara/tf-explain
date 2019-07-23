import shutil
from pathlib import Path

from tf_explain.callbacks.grad_cam import GradCAMCallback


def test_should_call_grad_cam_callback(random_data, convolutional_model):
    x, y = random_data
    convolutional_model.compile(optimizer="adam", loss="categorical_crossentropy")

    output_dir = Path("tests") / "test_logs"
    callbacks = [
        GradCAMCallback(
            validation_data=(x, None),
            layer_name="activation_1",
            class_index=0,
            output_dir=output_dir,
        )
    ]

    convolutional_model.fit(x, y, batch_size=2, epochs=2, callbacks=callbacks)

    assert len(list(output_dir.glob("*"))) == 2

    shutil.rmtree(output_dir)
