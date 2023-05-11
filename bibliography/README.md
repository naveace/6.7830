# bibliography
Machine learning bibliography

Purpose
=======

The purpose of this repository is to maintain an up-to-date list of references
to be linked to from papers we write. The idea is to a shared .bib file to be
used for every project, and have each paper point directly to it. 

Entry Conventions
=================

- when creating the keyword for a given citation, the format is 
<Lastname><Lastname_initial>...<Lastname_initial><YY><a|b|c|...>, where we use
a,b,c,... when collisions occur in the prefix.

For example,
~~~~
@InProceedings{SchneiderBH19a,
	Title		= {Deep{OBS}: A Deep Learning Optimizer Benchmark Suite},
	Author		= {Frank Schneider and Lukas Balles and Philipp Hennig},
	Booktitle	= {International Conference on Learning Representations (ICLR)},
	Year		= {2019}
}

@InProceedings{SchneiderBH19b,
	Title		= {Some other paper by the same authors},
	Author		= {Frank Schneider and Lukas Balles and Philipp Hennig},
	Booktitle	= {International Conference on Machine Learning (ICML)},
	Year		= {2019}
}
~~~~

- use a consistent format for the name of every conference, and don't include
  the year/iteration of the conference. For example, a good and currently
  consistent choice is 
~~~~
Booktitle = {International Conference on Learning Representations (ICLR)}
~~~~
whereas the following is discouraged:
~~~~
Booktitle = {Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security}
~~~~

- add an arXiv identifier for articles that are currently not in a conference
  proceedings, for example 
~~~~
booktitle = {ArXiv preprint arXiv:1807.07978}
~~~~

- in general, try to remove spurious/unnecessary information. The 'title',
  'author', 'booktitle' and 'year' fields are typically all we need.

How to use
==========

- follow the keyword conventions

- cite the conference where the reference appeared, if any (like in the
  example above).

- avoid duplicates by checking before you add.
