# PDTW-Plugin
### The Python implementation of paper 《An Algorithm for Improving Lower Bounds in Dynamic Time Warping》

The PDTW plugin improves the tightness of lower bounds by selectively replacing parts of existing lower bound computations with PDTW, resulting in additional pruning of the costly original DTW calculations. The following figure with a visualization of LB_Keogh and the PDTW plugin shows how this idea works.

![PDTW_Keogh_Seg1=2](https://github.com/user-attachments/assets/fcb8ae72-ef09-414c-8192-37971a1f4cfc)

We provide 4 well-known lower bounds and their enhanced versions by the PDTW plugin: LB_Keogh and LB_KP, LB_New and LB_NP, LB_Petitjean and LB_PP, LB_Webb and LB_WP.

### All related functions are in `Functions.py`, enhanced lower bounds are in `LB_KP.py`, `LB_NP.py`, `LB_PP.py`, and `LB_WP.py`.

Please read and run Quick_start.py to learn more about how functions work. The example time series data from dataset `CBF`. More datasets used in our experiments can be found at [UCR Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).

***

***If these codes can help you, please give a STAR.***

***If you use these codes or ideas in your research/software/product, please cite our paper:*** :+1:

`TBD`


#### Be sure your use follows the license.
