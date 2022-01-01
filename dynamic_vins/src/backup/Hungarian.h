//
// Created by chen on 2021/11/30.
//

#ifndef DYNAMIC_VINS_HUNGARIAN_H
#define DYNAMIC_VINS_HUNGARIAN_H



class HungarianAlgorithm {
public:
    double Solve(std::vector<std::vector<double>> &DistMatrix, std::vector<int> &Assignment);

private:
    void assignmentoptimal(int *assignment, double *cost, double *distMatrix, int nOfRows, int nOfColumns);

    void buildassignmentvector(int *assignment, bool *starMatrix, int nOfRows, int nOfColumns);

    void computeassignmentcost(int *assignment, double *cost, double *distMatrix, int nOfRows);

    void step2a(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
                bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);

    void step2b(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
                bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);

    void step3(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
               bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);

    void step4(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
               bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);

    void step5(int *assignment, double *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix,
               bool *coveredColumns, bool *coveredRows, int nOfRows, int nOfColumns, int minDim);
};



#endif //DYNAMIC_VINS_HUNGARIAN_H
