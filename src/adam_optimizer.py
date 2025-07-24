"""
ADAM Optimizer Implementation for Linear Regression

This implementation demonstrates the ADAM optimization algorithm applied to 
a simple linear regression problem. The visualization shows how the model
evolves over time as it learns to fit the data.

Author: [Your Name]
Project: ADAM Optimizer Portfolio Demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

def adam_optimizer(params, gradients, t, m, v, step_size=0.001, b_1=0.9, b_2=0.999, eps=10e-8):
    """Implementation of ADAM optimization algorithm"""
    t = t + 1
    g_t = gradients
    m_t = b_1 * m + (1 - b_1) * g_t
    v_t = b_2 * v + (1 - b_2) * g_t**2
    _m_t = m_t / (1 - b_1**t)
    _v_t = v_t / (1 - b_2**t)
    params = params - (step_size * _m_t) / (np.sqrt(_v_t) + eps)
    return params, m_t, v_t, t

def compute_gradients(x, y_true, y_pred, params):
    """Compute gradients of MSE loss with respect to linear regression parameters"""
    n = len(y_true)
    error = y_pred - y_true
    dL_dp0 = (2 / n) * np.sum(error * x)
    dL_dp1 = (2 / n) * np.sum(error)
    return np.array([dL_dp0, dL_dp1])

def linear_model(input_data, params):
    """Linear model: y = slope * x + intercept"""
    return input_data * params[0] + params[1]

def create_visualization_animation():
    """Create animated visualization of model learning process"""
    
    # Training data
    x_data = np.array([0, 1, 3, 5, 2, 8, 1])
    y_data = np.array([100, 200, 300, 400, 200, 900, 106])
    
    # Initialize model parameters [slope, intercept]
    params = np.array([0.2, 1.3])
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    
    # Storage for animation frames
    param_history = []
    loss_history = []
    prediction_history = []
    
    # Setup visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    line_pred, = ax1.plot([], [], 'r-', alpha=0.7, label='Model Prediction')
    ax1.scatter(x_data, y_data, c='blue', s=50, label='Training Data')
    ax1.set_xlim(-1, 9)
    ax1.set_ylim(0, 1000)
    ax1.set_xlabel('Input (x)')
    ax1.set_ylabel('Output (y)')
    ax1.set_title('Model Learning Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Loss Function Over Time')
    ax2.grid(True, alpha=0.3)
    line_loss, = ax2.plot([], [], 'g-')

    plt.tight_layout()

    # Training loop
    print("Training ADAM optimizer on linear regression problem...")
    for t in range(3_000):
        prediction = linear_model(x_data, params)
        loss = np.mean((prediction - y_data) ** 2)

        if t % 10 == 0:
            param_history.append(params.copy())
            loss_history.append(loss)
            prediction_history.append(prediction.copy())

        gradients = compute_gradients(x_data, y_data, prediction, params)
        params, m, v, t = adam_optimizer(params, gradients, t, m, v, step_size=0.1)

        if t % 500 == 0:
            print(f"Epoch {t:4d} | Loss: {loss:8.2f} | Parameters: [{params[0]:6.2f}, {params[1]:6.2f}]")

    def animate(frame):
        if frame < len(param_history):
            pred = prediction_history[frame]
            line_pred.set_data(x_data, pred)
            
            line_loss.set_data(range(0, frame*10, 10), loss_history[:frame])
            ax2.relim()
            ax2.autoscale_view()
            
            ax1.set_title(f'Model Learning - Iteration {frame*10}')
            
        return line_pred, line_loss

    print("\nGenerating animation...")
    anim = FuncAnimation(fig, animate, frames=len(param_history), 
                        interval=200, blit=False, repeat=True)
    
    return anim, fig, params, x_data, y_data

if __name__ == "__main__":
    # Create and display animation
    animation, figure, final_params, x_data, y_data = create_visualization_animation()
    plt.show()
    
    # Save animation to GitHub directory
    print("Saving animation to .github/adam_optimization.gif")
    os.makedirs('.github', exist_ok=True)
    animation.save('.github/adam_optimization.gif', writer='pillow', fps=5)
    
    # Display final results
    final_prediction = linear_model(x_data, final_params)
    final_loss = np.mean((final_prediction - y_data) ** 2)
    
    print("\n" + "="*50)
    print("FINAL MODEL RESULTS")
    print("="*50)
    print(f"Learned Parameters: slope = {final_params[0]:.2f}, intercept = {final_params[1]:.2f}")
    print(f"Final Mean Squared Error: {final_loss:.2f}")
    print("\nPredictions vs Actual:")
    for i in range(len(x_data)):
        print(f"  x={x_data[i]:1d} → Predicted: {final_prediction[i]:6.1f}, Actual: {y_data[i]:3d}")
    print("="*50) 
