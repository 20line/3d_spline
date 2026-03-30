# 3d_spline
Study of cubic spline approximation. Constructing a natural parametric cubic spline from a given set of points on a planar contour, calculating the spline coefficients, visualizing the result, and estimating the interpolation error.

1. 1_mondelbrot - This is the file for running the Mandelbrot set simulation. After running, a window appears where we randomly select a subset of the point set for further processing. After closing the window, the file "contour.txt" is saved with the coordinates of each point in the selected subset.

2. 1_parametr_t - Script for finding the t parameter.
We associate one t parameter with each pair of coordinates.
I used the accumulated arc length as the parameter—that is, the distance traveled from the beginning of the arc to the point with the current coordinates.
We can use it to calculate the x- and y-distance separately for each point to obtain a more accurate interpolation (approximation).

3. 2_builder.py - a simple script for visualizing the selected point set

4. 3_new_P.py - a script for selecting and visualizing interpolation nodes (the original set is called P, hence the file name)

5. 4_spline.py - a script for finding spline coefficients

6. 5_spline_error - calculates spline error

7. 6_visual.py - visualization of the entire work - the selected point set, nodes, and the spline
