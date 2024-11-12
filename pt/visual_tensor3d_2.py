import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches


def visualize_tensor_3d(tensor):
    """
    Visualize a 3D tensor as a 3D geometric representation with values on visible faces

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

                # Add value text on visible faces
                value = str(tensor[z, y, x])

                # Top face
                ax.text(x + 0.5, y + 0.5, z + 1,
                        value,
                        horizontalalignment='center',
                        verticalalignment='bottom')

                # Right face
                ax.text(x + 1, y + 0.5, z + 0.5,
                        value,
                        horizontalalignment='left',
                        verticalalignment='center')

                # Front face
                ax.text(x + 0.5, y, z + 0.5,
                        value,
                        horizontalalignment='center',
                        verticalalignment='center',
                        zdir='y')

    # Set axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Set integer ticks
    ax.set_xticks(np.arange(width + 1))
    ax.set_yticks(np.arange(height + 1))
    ax.set_zticks(np.arange(depth + 1))

    # Set axis limits with some padding
    ax.set_xlim(-0.2, width + 0.2)
    ax.set_ylim(-0.2, height + 0.2)
    ax.set_zlim(-0.2, depth + 0.2)

    # Adjust view angle for better visualization
    ax.view_init(elev=20, azim=45)

    # Add grid
    ax.grid(True)

    plt.title(f'3D Tensor Visualization ({depth}x{height}x{width})')
    plt.tight_layout()
    plt.show()


# Example usage with the tensor from the image
example_tensor = np.array([
    [[12, 13, 14, 15],
     [16, 17, 18, 19],
     [20, 21, 22, 23]],

    [[0, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9, 10, 11]]
])

visualize_tensor_3d(example_tensor)