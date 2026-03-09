# Signal Strength Visualizer

This project contains the rf visualizer made for assignment 3 in the course 5730.26 Wireless Communication.

The visualizer expects a floorplan image and an annotated grid. It is also necessary to specify the cell grid size e.g. 0.5 m, for the visualizer to work correctly.

## To run the visualizer:
```
pip install -r requirements.txt
python ./rf-visualize.py
```
To run with default settings.

To see available arguments run:

```
python ./rf-visualize.py --help
```

## To run the annotator
To run for a floorplan without a grid you must first annotate the floorplan with the annotator.

To run the annotator with default settings:
```
python ./floorplan-annotator.py
```

To see available arguments run:
```
python ./floorplan-annotator.py --help
```