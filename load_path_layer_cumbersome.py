# (Only under linux) the annotator can't load path annotations (path = not only start-end lines but several start-end lines)
# Too load a path layer type the following lines in the napari command line:

path_annot = np.load("annotations/faeden.npy", allow_pickle=True).item()

path_shapes = path_annot["shapes"]

faeden_layer = viewer.add_shapes(path_shapes, shape_type="path", edge_color="magenta", edge_width=40, opacity=0.1)




