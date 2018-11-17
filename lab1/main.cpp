#include <iostream>
#include <ctime>
#include <cmath>
#include <stdlib.h>
#include <iomanip>

typedef long double my_type;

using namespace std;

void showMatrix(my_type **t, int n, int m, int extraSize);
void showArray(my_type *t, int n);
my_type **readMatrix(my_type **t, int n, int m);
my_type **readArrayToMatrix(my_type **t, int n, int m);
my_type *gaussElimination(my_type **t, int n, int m);
my_type **generateMatrix1(my_type **t, int n, int m);
my_type **generateMatrix2(my_type **t, int n, int m);
my_type **generateMatrix3(my_type **t, int n);
my_type *readArray(my_type *p, int n);
my_type *generateArray(my_type *p, int n);
my_type *multiplyMatrix(my_type **A1, int n, int m, my_type *A2);
my_type *getSolutionFromMatrix(my_type **A, int n, int m);
void showDifferenceEu(my_type *t1, my_type *t2, int n);
void showDifferenceMax(my_type *t1, my_type *t2, int n);
my_type *solve3DiagonalMatrix(my_type *a, my_type *c, my_type *b, int n);
my_type *multiply3diagonalMatrix(my_type *a, my_type *c, my_type *x, int n);
my_type **generateMatrix3(my_type **t, int n, int m);

// shift significant places before comma, cut everything after comma
// then shift everything back again
my_type getPrecision(my_type value, my_type precision)
{
    return value;
    //return (floor((value * pow(10, precision) + 0.5)) / pow(10, precision));
}


int n = 1000;

const int prec = 13;     //significant places
const my_type eps = 1 / pow(10, prec);
int m;


int main() {

    // define size of the matrix
    //int n,m;
    cout << "Precision = " << eps << endl;
    cout << "A - matrix size: " << n <<endl;
    cout << sizeof(eps) * 8 * n << endl;
    //cin >> n;
    m = n;

    // memory allocation

    my_type **A = new my_type*[n];
    for(int i = 0; i<n ; i++){
        A[i] = new my_type[m+1];
    }

    my_type *x = new my_type[m];
    my_type *B = new my_type[m];


    // Multiplying
    //A = generateMatrix1(A, n, m);
    //A = generateMatrix2(A, n, m);
    A = generateMatrix3(A, n, m);

/*
    //trojdiagonal
    my_type k = 6.0;
    my_type *a = new my_type[n];
    my_type *c = new my_type[n];
    c[0] = 1.0 / (1 + m);
    for(int i=1; i<n-1;i++){
        a[i] = k / (i + 2 + m);
        c[i] = 1.0 / (i + 1 + m);
    }
    a[n-1] = k / (n + m + 1);

    //trojdiagonal end
*/

    x = generateArray(x, m);
    //showMatrix(A, n, m, 0);
    cout << "Multiply by (expected result) =============" << endl;
    //showArray(x, m);
    cout << "===========================================" << endl;
    B = multiplyMatrix(A, n, m, x);
    //B = multiply3diagonalMatrix(a, c, x, n);
    cout << "Result:" << endl;
    //showArray(B, m);
    cout<<endl;

    // only for testing
    // Resolving
    //A = readMatrix(A, n, m);
    //A = readArrayToMatrix(A, n, m);
    //or
    //generateMatrix(A, n, m);


    // copy the result to full A matrix -- only with multiplying part
    for(int i=0; i<m; i++){
        B[i] = getPrecision(B[i], prec);
        A[i][m] = B[i];
    }

    //showMatrix(A, n, m+1, 1);

    my_type *solution = gaussElimination(A, n, m+1);
    if(solution != NULL) showDifferenceEu(x, solution, m);
    if(solution != NULL) showDifferenceMax(x, solution, m);

    //my_type *solution3 = solve3DiagonalMatrix(a, c, B, n);
    //showDifferenceEu(x, solution3, n);
    //showDifferenceMax(x, solution3, n);

    return 0;
}


my_type **readArrayToMatrix(my_type **t, int n, int m){

    cout << "Array of size " << n << " (with spaces):" << endl;
    for(int i=0; i<n ; i++){
        cin >> t[i][m];
    }
    cout << endl;
    return t;
}


my_type **readMatrix(my_type **t, int n, int m){

    cout << "Matrix " << n << " x " << m << " (with spaces):" << endl;
    for(int i=0; i<n ; i++){
        for(int j=0; j<m; j++){
            cin >> t[i][j];
        }
    }
    cout << endl;
    return t;
}

// first task
my_type **generateMatrix1(my_type **t, int n, int m){

    for(int i=0; i<n ; i++){
        for(int j=0; j<m; j++){
            if(i == 0)  // i == 1, dif notation
                t[i][j] = 1.0;
            else
                t[i][j] = 1.0 / ((i+1) + (j+1) - 1);    // diffrent notation from 0 and from 1
            t[i][j] = getPrecision(t[i][j], prec);
        }
    }
    cout << endl;
    return t;
}


// second task
my_type **generateMatrix2(my_type **t, int n, int m){

    for(int i=0; i<n ; i++){
        for(int j=0; j<m; j++){
            if(j >= i)
                t[i][j] = my_type(2 * (i+1)) / (j+1);
            else
                t[i][j] = t[j][i];
            t[i][j] = getPrecision(t[i][j], prec);
        }
    }
    cout << endl;
    return t;
}

// third task
my_type **generateMatrix3(my_type **t, int n, int m){

    for(int i=0; i<n ; i++){
        for(int j=0; j<m; j++){
            if(j == i)
                t[i][j] = my_type(6);
            else if(i+1 == j){
                t[i][j] = 6.0/(i+1 + 2);
            } else if(i-1 == j){
                t[i][j] = 6.0/(i+1 + 2 + 1);
            } else
                t[i][j] = 0.0;
            t[i][j] = getPrecision(t[i][j], prec);
        }
    }
    cout << endl;
    return t;
}




void showMatrix(my_type **t, int n, int m, int extraSize = 0){

    for(int i=0; i<n ; i++){
        for(int j=0; j<m; j++){
            cout << setprecision(16) << t[i][j] << "\t";
            if(j+2 == m && extraSize == 1) cout<< "|" << "\t";
        }
        cout << endl;
    }
    cout << endl;
}


void showArray(my_type *t, int n){

    for(int i=0; i<n ; i++){
        cout << setprecision(17) << t[i] << "\t\t";
    }
    cout << endl;
}


my_type *gaussElimination(my_type **t, int n, int m){
    my_type factor;
    int factor_column = 0;
    int factor_line = 0;
    int range = m-1;
    int conflict = 0;
    int identity = 0;

    clock_t t_begin = clock();

    // i = step number
    for(int i=1; i<n; i++){

        // k - multiplies every line
        for(int k = i; k<n; k++){

            if(t[factor_line][factor_column] == 0 ){
                // search for non zero element
                int l;
                for(l=k; l<n; l++){
                    if(t[l][factor_column] != 0) break;
                }

                if(l<n){
                    //element was found
                    my_type *tmp = t[factor_line];
                    t[factor_line] = t[l];
                    t[l] = tmp;
                    showMatrix(t, n, m, 1);
                } else {
                    factor_column++;
                    continue;
                }
            }
            factor = getPrecision(t[k][factor_column] / t[factor_line][factor_column], prec);

            // j - subtract every column in k line
            int sum = 0;
            for(int j=factor_column; j<m; j++){
                t[k][j] = getPrecision(t[k][j] - factor * t[factor_line][j], prec);
                sum += getPrecision(t[k][j], prec);
            }
            if(sum - t[k][m-1] == 0) range--;
            if(sum - t[k][m-1] == 0 && t[k][m-1] != 0 ) conflict++;
            if(sum - t[k][m-1] == 0 && t[k][m-1] == 0) identity++;
        }
        factor_column++;
        factor_line++;
    }


    clock_t t_end = clock();
    my_type elapsed_secs = my_type(t_end - t_begin) / CLOCKS_PER_SEC;
    cout << "********** Elapsed seconds: " << elapsed_secs << endl;

    cout << "Rzad macierzy glownej wynosi " << range << endl;
    if(conflict != 0) cout << "Uklad sprzeczny!!!" << endl;
    else if(identity != 0) cout << "Uklad tozsamosciowy!!!" << endl;
    else {
        my_type *solution = getSolutionFromMatrix(t, n, m);
        cout << "============== ROZWIAZANIE ===============" << endl;
        //showArray(solution, n);
        cout << "==========================================" << endl;
        return solution;
    }
    return NULL;
}


my_type *multiplyMatrix(my_type **A1, int n, int m, my_type *A2){
    my_type *p = new my_type[n];

    for(int i=0 ; i<n; i++){
        p[i] = 0;
        for(int j=0; j<m; j++){
            p[i] += getPrecision(A1[i][j] * A2[j], prec);
        }
    }
    return p;
}


my_type *readArray(my_type *p, int n){
    cout << "Array x to multiply: " << endl;
    for(int i=0; i<n ; i++){
        cin >> p[i];
    }
    cout << endl;
    return p;
}

my_type *generateArray(my_type *p, int n){
    cout << "Array x to multiply: " << endl;
    for(int i=0; i<n ; i++){
        if(i%2 == 0)
            p[i] = 1.0;
        else
            p[i] = -1.0;
    }
    cout << endl;
    return p;
}

my_type *getSolutionFromMatrix(my_type **A, int n, int m){
    my_type *solution = new my_type[n];
    for(int i=0; i<n; i++) solution[i] = 0;

    for(int i=n-1; i>=0; i--){
        solution[i] = A[i][m-1];
        for(int j=m-2; j > i; j--){
            solution[i] -= A[i][j] * solution[j];
        }
        solution[i] /= A[i][i];
    }
    return solution;
}


void showDifferenceEu(my_type *t1, my_type *t2, int n){
    my_type *dif = new my_type[n];
    my_type res = 0;

    cout << endl << "======== Euclides difference: ===========" << endl;
    for(int i=0; i<n; i++){
        dif[i] = t2[i] - t1[i];
        //cout << dif[i] <<endl;
        res += dif[i] * dif[i];
    }
    res = sqrt(res);
    cout << setprecision(16) << res << "\t";
    cout << endl << "=========================================" << endl;
}

void showDifferenceMax(my_type *t1, my_type *t2, int n){

    cout << endl << "======== Maximum difference: ===========" << endl;
    my_type maxi = 0;
    my_type tmp;
    for(int i=0; i<n; i++){
        tmp = abs(t2[i] - t1[i]);
        //if(tmp != 0)
        //    cout << "diff\t" << tmp << endl;
        if(tmp > maxi)
            maxi = tmp;
    }
    //cout << "TEST: " << setprecision(16) << t2[100] - t1[100] <<endl;
    cout << setprecision(16) << maxi;
    cout << endl << "=========================================" << endl;
}



my_type *solve3DiagonalMatrix(my_type *a, my_type *c, my_type *b, int n){

    my_type *l = new my_type[n];
    my_type *u = new my_type[n];
    my_type *x = new my_type[n];
    my_type *y = new my_type[n];

    my_type k = 6.0;
    my_type m = 2.0;

    u[0] = k;
    //c[0] = 1.0 / (1 + m);
    for(int i=1; i<n-1;i++){
        //a[i] = k / (i + 2 + m);
        //c[i] = 1.0 / (i + 1 + m);

        l[i] = (my_type)(a[i] / u[i-1]);
        u[i] = k - (l[i] * c[i-1]);
    }
    //a[n-1] = k / (n + m + 1);
    l[n-1] = (my_type)(a[n-1] / u[n-2]);
    u[n-1] = k - (l[n-1] * c[n-2]);


    // L * y = b
    // U * x = y

    clock_t t_begin = clock();

    y[0] = b[0];
    for(int i=1; i<n; i++){
        y[i] = (my_type)(b[i] - (l[i] * y[i-1]));
    }

    x[n-1] = (my_type)(y[n-1] / u[n-1]);
    for(int i=n-2; i>=0 ; i--){
        x[i] = (my_type)((y[i] - (c[i] * x[i+1])) / u[i]);
    }

    clock_t t_end = clock();
    my_type elapsed_secs = my_type(t_end - t_begin) / CLOCKS_PER_SEC;
    cout << "********** Elapsed seconds: " << elapsed_secs << endl;

    cout << "============== ROZWIAZANIE2 ==============" << endl;
    //showArray(x, n);
    cout << "==========================================" << endl;
    return x;
}



my_type *multiply3diagonalMatrix(my_type *a, my_type *c, my_type *x, int n){
    my_type *b = new my_type[n];
    my_type k = 6.0;

    b[0] = k * x[0];
    b[0] += c[0] * x[1];
    for(int i=1 ; i<n-1; i++){
        b[i] = a[i] * x[i-1];
        b[i] += k * x[i];
        b[i] += c[i] * x[i+1];
    }
    b[n-1] = a[n-1] * x[n-2];
    b[n-1] += k * x[n-1];
    return b;
}
