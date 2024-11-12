import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


def visualize_tensor_3d(tensor):
    """
    Visualize a 3D tensor as a 3D geometric representation with clear visible values on faces

    Args:
        tensor: A 3D numpy array
    """
    # Get tensor dimensions
    depth, height, width = tensor.shape

    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define the vertices of each cube in the tensor
    def get_cube_vertices(x, y, z):
        vertices = np.array([
            [x, y, z], [x + 1, y, z], [x + 1, y + 1, z], [x, y + 1, z],
            [x, y, z + 1], [x + 1, y, z + 1], [x + 1, y + 1, z + 1], [x, y + 1, z + 1]
        ])
        return vertices

    # Define faces for each cube
    def get_cube_faces(vertices):
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]]  # right
        ]
        return faces

    def add_text_with_background(ax, x, y, z, text, ha='center', va='center', zdir=None):
        """Helper function to add text with white background"""
        if zdir is None:
            ax.text(x, y, z, text,
                    horizontalalignment=ha,
                    verticalalignment=va,
                    color='black',
                    fontsize=12,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        else:
            ax.text(x, y, z, text,
                    horizontalalignment=ha,
                    verticalalignment=va,
                    color='black',
                    fontsize=12,
                    fontweight='bold',
                    zdir=zdir,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    # Plot from back to front to handle overlapping correctly
    for z in reversed(range(depth)):
        for y in reversed(range(height)):
            for x in reversed(range(width)):
                vertices = get_cube_vertices(x, y, z)
                faces = get_cube_faces(vertices)

                # Create nearly opaque cube
                cube = Poly3DCollection(faces, alpha=0.9)
                cube.set_facecolor('lightgray')
                cube.set_edgecolor('black')
                ax.add_collection3d(cube)

                # Add value text on visible faces with white background
                value = str(tensor[z, y, x])

                # Top face
                add_text_with_background(ax, x + 0.5, y + 0.5, z + 1,
                                         value,
                                         ha='center',
                                         va='bottom')

                # Right face
                add_text_with_background(ax, x + 1, y + 0.5, z + 0.5,
                                         value,
                                         ha='left',
                                         va='center')

                # Front face
                add_text_with_background(ax, x + 0.5, y, z + 0.5,
                                         value,
                                         ha='center',
                                         va='center',
                                         zdir='y')

    # Set axis labels with increased font size
    ax.set_xlabel('x', fontsize=12, labelpad=10)
    ax.set_ylabel('y', fontsize=12, labelpad=10)
    ax.set_zlabel('z', fontsize=12, labelpad=10)

    # Set axis limits with some padding
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(-0.2, height + 0.2)
    ax.set_zlim(-0.2, depth + 0.2)

    # Adjust view angle for better visualization
    ax.view_init(elev=20, azim=45)

    # Add grid with custom style
    ax.grid(True, linestyle='--', alpha=0.6)

    # Set title with increased font size
    plt.title(f'3D Tensor Visualization ({depth}x{height}x{width})', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


# Example usage with the tensor from the image
example_tensor = np.array([
    [[8, 9, 10, 11],
     [20, 21, 22, 23]],

    [[4, 5, 6, 7],
     [8, 9, 10, 11]],

    [[0, 1, 2, 3],
     [4, 5, 6, 7]]
])

visualize_tensor_3d(example_tensor)
