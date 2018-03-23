# Selected Topics in Soft Computing: Final Project

This is the code for my final project for the course [Selected Topics in Soft
Computing](https://korppi.jyu.fi/kotka/course/student/courseInfo.jsp?course=216250) at JYU.  The
project consists of replicating the paper "Application of genetic programming to induction of linear
classification trees" (Bot and Langdon, 2000).

## Data

The `data` folder contains the following three data sets from the UCI Machine Learning Repository,
which are used for evaluation in the paper:

- [Glass identification](https://archive.ics.uci.edu/ml/datasets/glass+identification)
- [Ionosphere](https://archive.ics.uci.edu/ml/datasets/ionosphere)
- [Image segmentation](https://archive.ics.uci.edu/ml/datasets/image+segmentation) (Only the training
  set was used.)

(The Pima indians diabetes data set apperantly has been removed due to license restrictions.)

These datasets were retrieved through Jack Dunn's
[`uci-data`](https://github.com/JackDunnNZ/uci-data) script.

Additionally, there is an artificial two-dimensional, linearly separable data set `testdata.data`,
which I found on [Github](https://github.com/cuekoo/Binary-classification-dataset) and used for
testing the implementation.

## Boltzmann Sampler

Recursively generating random trees can be problematic if done naively, since the resulting samples
are usually either very small or very large.  Out of interest, and deviating from the paper (and the
original description in (Montana, 1995)), I translated Brent Yorgey's [size-limited critical
Boltzmann
sampler](https://byorgey.wordpress.com/2013/04/25/random-binary-trees-with-a-size-limited-critical-boltzmann-sampler-2/),
which allows to efficiently generate random trees in a given size range.

## References

```
@article{montana_strongly_1995,
	title = {Strongly typed genetic programming},
	volume = {3},
	url = {http://davidmontana.net/papers/stgp.pdf},
	doi = {10.1162/evco.1995.3.2.199},
	pages = {199--230},
	number = {2},
	journaltitle = {Evolutionary computation},
	author = {Montana, David J.},
	date = {1995}
}

@inproceedings{bot_application_2000,
	title = {Application of genetic programming to induction of linear classification trees},
	url = {http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.35.8630&rep=rep1&type=pdf},
	doi = {10.1007/978-3-540-46239-2_18},
	pages = {247--258},
	booktitle = {European Conference on Genetic Programming},
	publisher = {Springer},
	author = {Bot, Martijn {CJ} and Langdon, William B.},
	date = {2000}
}

@misc{dua_2017,
    author = {Dheeru, Dua and Karra Taniskidou, Efi},
    year = {2017},
    title = {{UCI} Machine Learning Repository},
    url = {http://archive.ics.uci.edu/ml},
    institution = {University of California, Irvine, School of Information and Computer Sciences} 
} 


@thesis{qureshi_evolution_2001,
	location = {London},
	title = {The Evolution of Agents},
	url = {https://www.cl.cam.ac.uk/~jac22/otalks/aqureshi-draft.pdf},
	institution = {University of London},
	author = {Qureshi, Mohammad Adil},
	urldate = {2018-03-23},
	date = {2001}
}
```
