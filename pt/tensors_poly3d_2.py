import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def create_tensor_visualization(tensor_shape=(2, 3, 4)):
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define the dimensions
    depth, height, width = tensor_shape

    # Create the coordinates for each point
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                # Calculate the index
                index = z * (height * width) + y * width + x

                # Invert z-coordinate for downward positive direction
                z_inv = depth - 1 - z

                # Draw the cube at (x, y, z)
                # Define the vertices of the cube with inverted z
                vertices = np.array([
                    [x, y, z_inv], [x + 1, y, z_inv], [x + 1, y + 1, z_inv], [x, y + 1, z_inv],
                    [x, y, z_inv + 1], [x + 1, y, z_inv + 1], [x + 1, y + 1, z_inv + 1], [x, y + 1, z_inv + 1]
                ])

                # Define the faces of the cube
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                    [vertices[0], vertices[3], vertices[7], vertices[4]]  # left
                ]

                # Create the 3D collection
                pc = Poly3DCollection(faces, alpha=0.25, edgecolor='black')
                pc.set_facecolor('lightgray')
                ax.add_collection3d(pc)

                # Add the index text in the center of the cube
                center = vertices.mean(axis=0)
                # Change text color to dark blue and increase font weight
                ax.text(center[0], center[1], center[2], str(index),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='navy',
                        fontweight='bold',
                        fontsize=10)

    # Set the axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set integer ticks
    ax.set_xticks(np.arange(width + 1))
    ax.set_yticks(np.arange(height + 1))
    ax.set_zticks(np.arange(depth + 1))

    # Invert z-axis ticks to match the direction
    ax.set_zticks(np.arange(depth + 1))
    ax.set_zticklabels(np.arange(depth + 1)[::-1])

    # Set the viewing angle
    ax.view_init(elev=20, azim=45)

    # Set the axis limits
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_zlim(-0.1, depth + 0.1)  # Added small padding

    plt.title('3D Tensor Visualization')
    plt.tight_layout()
    plt.show()


# Create the visualization with a 2x3x4 tensor
create_tensor_visualization((2, 3, 4))
