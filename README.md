# Share Arithmetic Expressions for Survival Analysis 
This repo extends SHAREs, from the paper [*Shape Arithmetic Expressions: Advancing Scientific Discovery Beyond Closed-form Equations*](https://proceedings.mlr.press/v238/kacprzyk24a/kacprzyk24a.pdf), to survival regression tasks. 

## Clone the repository
Clone the repository using

```
git clone --recurse-submodules -j8 https://github.com/steliosbl/SurvSHAREs.git
```

## Dependencies
You can install all required dependencies using conda and the following command
```
conda env create -n survshares --file environment.yml
```

## References
This code is an adaptation of *Kacprzyk, K. & van der Schaar, M. Shape Arithmetic Expressions: Advancing Scientific Discovery Beyond Closed-form Equations. in Proceedings of The 27th International Conference on Artificial Intelligence and Statistics (PMLR, 2024).* [Publication Link](https://proceedings.mlr.press/v238/kacprzyk24a/kacprzyk24a.pdf). [Original Repo](https://github.com/krzysztof-kacprzyk/SHAREs).

This code uses a modified instance of [gplearn](https://github.com/trevorstephens/gplearn).
