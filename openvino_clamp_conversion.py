import numpy as np
import tensorflow as tf
import openvino as ov
from openvino.tools.mo import convert_model
from openvino.runtime import Core, PartialShape

def create_model_with_clamp():
    """Create a simple TensorFlow model with clamping functionality"""
    input_layer = tf.keras.layers.Input(shape=(224, 224, 1), name='input')
    # Add a custom layer that implements clamping
    x = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -40, 40))(input_layer)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs=input_layer, outputs=output)

def convert_to_openvino_with_clamp(model_path, min_value, max_value):
    """
    Convert a model to OpenVINO IR format with clamping functionality
    
    Args:
        model_path: Path to the input model
        min_value: Minimum value for clamping
        max_value: Maximum value for clamping
    """
    # Load the model
    core = Core()
    model = core.read_model(model_path)
    
    # Add preprocessing steps
    p = ov.preprocess.PrePostProcessor(model)
    
    # Get input information
    input_info = model.inputs[0]
    input_name = input_info.get_any_name()
    
    # Add clamping as preprocessing step
    p.input(input_name).preprocess().custom(
        lambda data: np.clip(data, min_value, max_value)
    )
    
    # Apply preprocessing
    model = p.build()
    
    # Save the model
    ov.save_model(model, "model_with_clamp.xml")
    return model

def main():
    # Create and save a sample model
    model = create_model_with_clamp()
    model.save("sample_model.h5")
    
    # Convert with clamping
    min_temp = -40  # Example minimum temperature
    max_temp = 40   # Example maximum temperature
    
    converted_model = convert_to_openvino_with_clamp(
        "sample_model.h5",
        min_temp,
        max_temp
    )
    
    print("Model converted successfully with clamping functionality!")
    print(f"Clamping range: [{min_temp}, {max_temp}]")

if __name__ == "__main__":
    main() 