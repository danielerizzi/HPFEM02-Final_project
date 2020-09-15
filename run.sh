#!/usr/bin/bash
# to compile and run: ./run.sh -c
# to just run       : ./run.sh

PROJECT_NAME="teslavalve"

## if -c is provided change the project in cmakelists to $PROJECT_NAME
[ "$1" == "-c" ] &&
sed -i "s/\(set[(]PROJECT_NAME \)\(.*\)\([)]\)/\1 $PROJECT_NAME\3/g" CMakeLists.txt

## if -c is provided compile the ufl file
count=`ls -1 *.ufl 2>/dev/null | wc -l`
[ "$1" == "-c" ] && [ $count != 0 ] && 
ffc -l dolfin *.ufl

## if -c is provided delete the old build folder
[ "$1" == "-c" ] &&
[ -d "build" ] && rm -rf build 

## if -c is provided make build directory, and compile the code
[ "$1" == "-c" ] &&
mkdir build && cd build && cmake .. && make && cd ..

## delete results folder if exists
[ -d "results" ] && rm -rf results

## move results to a different folder
# [ -d "results" ] && mv results results.old

## make results folder and run the executable
mkdir results && cd results
echo "localhost slots=10" > hostfile
export OMP_NUM_THREADS=1
mpirun --hostfile hostfile -n 6 ../build/$PROJECT_NAME
