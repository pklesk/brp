echo off
rem -----------------------------------------------------------------------------------------------------------------
rem help information about experiments runner
rem -----------------------------------------------------------------------------------------------------------------
python brp_paper_python_experiments.py -h

rem -----------------------------------------------------------------------------------------------------------------
rem example executions for the first data set ''normals 2D (small)'' and solvers: libsvm, liblinear, cvxopt
rem -----------------------------------------------------------------------------------------------------------------
python brp_paper_python_experiments.py normals2d_small libsvm -repetitions 100
python brp_paper_python_experiments.py normals2d_small liblinear -repetitions 100
python brp_paper_python_experiments.py normals2d_small cvxopt -repetitions 100

rem -----------------------------------------------------------------------------------------------------------------
rem example executions for the first data set ''normals 2D (small)'' and solvers: brp, brp_fast
rem remark (!): for these solvers python implementations are slower than Mathematica's C compiled ones (recommended)
rem -----------------------------------------------------------------------------------------------------------------
python brp_paper_python_experiments.py normals2d_small brp -T 100 -repetitions 100
python brp_paper_python_experiments.py normals2d_small brp -T 1000 -repetitions 100
python brp_paper_python_experiments.py normals2d_small brp -T 10000 -repetitions 100
python brp_paper_python_experiments.py normals2d_small brp -T 100000 -repetitions 10
python brp_paper_python_experiments.py normals2d_small brp_fast -T 1000 -T0 100 -alpha 0.05 -repetitions 10
python brp_paper_python_experiments.py normals2d_small brp_fast -T 10000 -T0 100 -alpha 0.05 -repetitions 10
python brp_paper_python_experiments.py normals2d_small brp_fast -T 100000 -T0 100 -alpha 0.05 -repetitions 10