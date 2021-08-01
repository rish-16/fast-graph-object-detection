# fast-graph-object-detection
Fast Object Detection with Graph Neural Networks

## What is this?
Object Detection with Graph Neural Networks is an idea I wanted to experiment with. It treats object detection tasks as node classification tasks where each node is a pixel in the image. 

## How does it work?
An image can be considered a grid graph where each pixel is an individual node. A neighbourhood is thus formed with 8 adjacent nodes and a central node. Node features can be the pixel value while node labels can be which object the pixel belongs to (for example, one of the 1000 categories from the MS COCO dataset). Creating this adjacency list is trivial.

Essentially, I reduce object detection to a node classification problem. 

## Variants
Instead of treating each pixel as a node, I also experiment with image patches. The `HxW` image is broken into non-overlapping image patches of size `PxP`. Each patch is considered a single node.

## Challenges
This method does not scale well when the image is very large owing to the number of pixel (nodes) to be trained upon.

## TODO
- [ ] Collate data into single source JSON
- [ ] Get segmentation masks
- [ ] Create graph dataset
    - [ ] Adjacency Matrix
    - [ ] Node labels based on Convex Hull
    - [ ] Reformat into `torch_geometric` standard
    - [ ] Feed into `torch_geometric.data.DataLoader`
- [ ] Build, train, and test GCN