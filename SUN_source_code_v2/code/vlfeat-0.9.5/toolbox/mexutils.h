/** @file    mexutils.h
 ** @brief   MEX utilities
 ** @author  Andrea Vedaldi
 **/

/* AUTORIGHTS
Copyright (C) 2007-09 Andrea Vedaldi and Brian Fulkerson

This file is part of VLFeat, available in the terms of the GNU
General Public License version 2.
*/

#ifndef MEXUTILS_H
#define MEXUTILS_H

#include"mex.h"
#include<vl/generic.h>
#include<ctype.h>
#include<stdio.h>
#include<stdarg.h>

#ifdef VL_COMPILER_MSC
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#endif

#if ! defined (MX_API_VER) | (MX_API_VER < 0x07030000)
typedef vl_uint32 mwSize ;
typedef vl_int32 mwIndex ;
#endif

/** @file mexutils.h

 This header file provides helper functions for writing MATLAB MEX
 files.

 - @ref mexutils-env "VLFeat environment"
 - @ref mexutils-array-test "Array tests"
 - @ref mexutils-options "Parsing options"

 @section mexutils-env VLFeat environment

 When the VLFeat DLL is linked to a MATLAB MEX files, at run time the
 MEX file must configure VLFeat to use MATLAB memory allocation and
 logging functions. This can be obtained by calling the macro
 ::VL_USE_MATLAB_ENV as the first line of each MEX file which is
 linked to the VLFeat DLL.

 @section mexutils-array-test Array tests

 MATLAB supports a variety of array types. Most MEX file arguments are
 restricted to a few types and must be properly checked at run time.
 ::mexutils.h provides some helper functions to make it simpler to
 check such arguments. MATLAB basic array types are:

 - Numeric array:
   @c mxDOUBLE_CLASS, @c mxSINGLE_CLASS,
   @c mxINT8_CLASS, @c mxUINT8_CLASS,
   @c mxINT16_CLASS, @c mxUINT16_CLASS,
   @c mxINT32_CLASS, @c mxUINT32_CLASS. Moreover:
   - all such types have a @e real component
   - all such types may have a @e imaginary component
   - @c mxDOUBLE_LCASS arrays with two dimensions can be @e sparse.
 - Logical array (@c mxLOGICAL_CLASS).
 - Character array (@c mxCHAR_CLASS).

 The other MATLAB array types are:

 - Struct array (@c mxSTRUCT_CLASS).
 - Cell array (@c mxCELL_CLASS).
 - Custom class array (@c mxCLASS_CLASS).
 - Unkown type array (@c mxUNKNOWN_CLASS).

 VLFeat defines a number of common classes of arrays and corresponding
 tests.

 - <b>Scalar array</b> is a non-sparse array with exactly one element.
   Note that the array may have an arbitrary number of dimensions, and
   be of any numeric or other type. All dimensions are singleton
   (which is implied by having exactly one element). Use ::vlmxIsScalar
   to test if an array is scalar.

 - <b>Vector array</b> is a non-sparse array which is either empty
   (empty vector) or has at most one non-singleton dimension. The
   array can be of any numeric or other type. The elements of such a
   MATLAB array are stored as a plain C array with a number of
   elements equal to the number of elements in the array (obtained
   with @c mxGetNumberOfElements). Use ::vlmxIsVector to test if an
   array is a vector.

 - <b>Matrix array</b> is a non-sparse array for which all dimensions
   beyond the first two are singleton, or a non-sparse array which is
   empty and for which at least one of the first two dimensions is
   zero. The array can be of any numeric or other type.  The
   non-singleton dimensions can be zero (empty matrix), one, or
   more. The element of such a MATLAB array are stored as a C array in
   column major order and its dimensions can be obtained by @c mxGetM
   and @c mxGetN.  Use ::vlmxIsMatrix to test if an array is a mtarix.

 - <b>Real array</b> is a numeric array (as for @c mxIsNumeric)
   without a complex component. Use ::vlmxIsReal to check if an array
   is real.

 - Use ::vlmxIsOfClass to check if an array is of a prescribed
   (storage) class, such as @c mxDOUBLE_CLASS.

 - <b>Plain scalar, vector, and matrix</b> are a scalar, vector, and
   matrix arrays which are <em>real</em> and of class @c
   mxDOUBLE_CLASS.  Use ::vlmxIsPlainScalar, ::vlmxIsPlainVector and
   ::vlmxIsPlainMatrix to check this.

 @section mexutils-options Parsing options

 It is common to pass optional arguments to a MEX file as option
 type-value pairs. Here type is a string identifying the option and
 value is a MATLAB array specifing its value. The function
 ::uNextOption can be used to simplify parsing a list of such
 arguments (similar to UNIX @c getopt). The functions ::mxuError
 and ::mxuWarning are shortcuts to specify VLFeat formatted errors.

 **/

/** @name Check for array attributes
 ** @{ */

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is of a prescribed class
 ** @param array MATLAB array.
 ** @param classId prescribed class of the array.
 ** @return ::VL_TRUE if the class is of the array is of the prescribed class.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsOfClass (mxArray const * array, mxClassID classId)
{
  return mxGetClassID (array) == classId ;
}

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is real
 ** @param array MATLAB array.
 ** @return ::VL_TRUE if the array is real.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsReal (mxArray const * array)
{
  return mxIsNumeric (array) && ! mxIsComplex (array) ;
}

/** @} */

/** @name Check for scalar, vector and matrix arrays
 ** @{ */

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is scalar
 ** @param array MATLAB array.
 ** @return ::VL_TRUE if the array is scalar.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsScalar (mxArray const * array)
{
  return (! mxIsSparse (array)) && (mxGetNumberOfElements (array) == 1)  ;
}

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is a vector.
 ** @param array MATLAB array.
 ** @param numElements number of elements (negative for any).
 ** @return ::VL_TRUE if the array is a vecotr of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsVector (mxArray const * array, vl_index numElements)
{
  mwSize numDimensions = mxGetNumberOfDimensions (array) ;
  mwSize const * dimensions = mxGetDimensions (array) ;
  vl_uindex di ;

  /* check that it is not sparse */
  if (mxIsSparse (array)) {
    return VL_FALSE ;
  }

  /* check that the number of elements is the prescribed one */
  if ((numElements >= 0) && (mxGetNumberOfElements (array) != numElements)) {
    return VL_FALSE ;
  }

  /* ok if empty */
  if (mxGetNumberOfElements (array) == 0) {
    return VL_TRUE ;
  }

  /* find first non-singleton dimension */
  for (di = 0 ; (dimensions[di] == 1) && di < numDimensions ; ++ di) ;

  /* skip it */
  if (di < numDimensions) ++ di ;

  /* find next non-singleton dimension */
  for (; (dimensions[di] == 1) && di < numDimensions ; ++ di) ;

  /* if none found, then ok */
  return di == numDimensions ;
}

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is a matrix.
 ** @param array MATLAB array.
 ** @param M number of rows (negative for any).
 ** @param N number of columns (negative for any).
 ** @return ::VL_TRUE if the array is a matrix of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsMatrix (mxArray const * array, vl_index M, vl_index N)
{
  mwSize numDimensions = mxGetNumberOfDimensions (array) ;
  mwSize const * dimensions = mxGetDimensions (array) ;
  vl_uindex di ;

  /* check that it is not sparse */
  if (mxIsSparse (array)) {
    return VL_FALSE ;
  }

  /* check that the number of elements is the prescribed one */
  if ((M >= 0) && (mxGetM (array) != M)) {
    return VL_FALSE;
  }
  if ((N >= 0) && (mxGetN (array) != N)) {
    return VL_FALSE;
  }

  /* ok if empty and either M = 0 or N = 0 */
  if ((mxGetNumberOfElements (array) == 0) && (mxGetM (array) == 0 || mxGetN (array) == 0)) {
    return VL_TRUE ;
  }

  /* ok if any dimension beyond the first two is singleton */
  for (di = 2 ; (dimensions[di] == 1) && di < numDimensions ; ++ di) ;
  return di == numDimensions ;
}
/** @} */

/** @name Check for plain arrays
 ** @{ */


/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is plain scalar
 ** @param array MATLAB array.
 ** @return ::VL_TRUE if the array is plain scalar.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsPlainScalar (mxArray const * array)
{
  return
  vlmxIsReal (array) &&
  vlmxIsOfClass(array, mxDOUBLE_CLASS)  &&
  vlmxIsScalar (array) ;
}

/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is a plain vector.
 ** @param array MATLAB array.
 ** @param numElements number of elements (negative for any).
 ** @return ::VL_TRUE if the array is a plain vecotr of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsPlainVector
(mxArray const * array, vl_index numElements)
{
  return
  vlmxIsReal (array) &&
  vlmxIsOfClass (array, mxDOUBLE_CLASS) &&
  vlmxIsVector (array, numElements) ;
}


/** ------------------------------------------------------------------
 ** @brief Check if a MATLAB array is a plain matrix.
 ** @param array MATLAB array.
 ** @param M number of rows (negative for any).
 ** @param N number of columns (negative for any).
 ** @return ::VL_TRUE if the array is a plain matrix of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE vl_bool
vlmxIsPlainMatrix (mxArray const * array, vl_index M, vl_index N)
{
  return
  vlmxIsReal (array) &&
  vlmxIsOfClass (array, mxDOUBLE_CLASS) &&
  vlmxIsMatrix (array, M, N) ;
}

/** @} */

/** ------------------------------------------------------------------
 ** @brief Let VLFeat use MATLAB memory allocation/logging facilities
 **
 ** This makes VLFeat use MATLAB version of the memory allocation and
 ** logging functions.
 **
 **/
#define VL_USE_MATLAB_ENV                                       \
  vl_set_alloc_func (mxMalloc, mxRealloc, mxCalloc, mxFree) ;   \
  vl_set_printf_func (mexPrintf) ;

/** ------------------------------------------------------------------
 ** @brief Create array with pre-allocated data
 **
 ** @param ndim    number of dimensions.
 ** @param dims    dimensions.
 ** @param classid storage class ID.
 ** @param data    pre-allocated data.
 **
 ** If @a data is set to NULL, the data is allocated from the heap.
 ** If @a data is a buffer allocated by @a mxMalloc, then this buffer
 ** is used as data.
 **
 ** @return new array.
 **/

static mxArray *
uCreateNumericArray (mwSize ndim, const mwSize * dims,
                     mxClassID classid, void * data)
{
  mxArray *A ;

  if  (data) {
    mwSize dims_ [2] = {0, 0} ;
    A = mxCreateNumericArray (2, dims_, classid, mxREAL) ;
    mxSetData (A, data) ;
    mxSetDimensions (A, dims, ndim) ;
  } else {
    A = mxCreateNumericArray (ndim, dims, classid, mxREAL) ;
  }

  return A ;
}

/** ------------------------------------------------------------------
 ** @brief Create an array with pre-allocated data
 **
 ** @param M       number of rows.
 ** @param N       number of columns.
 ** @param classid class ID.
 ** @param data    pre-allocated data.
 **
 ** If @a data is set to NULL, the data is allocated from the heap.
 ** If @a data is a buffer allocated by @a mxMalloc, then this buffer
 ** is used as data.
 **
 ** @return new array.
 **/

static mxArray *
uCreateNumericMatrix (int M, int N, mxClassID classid, void * data)
{
  mxArray *A ;

  if  (data) {
    A = mxCreateNumericMatrix (0, 0, classid, mxREAL) ;
    mxSetData (A, data) ;
    mxSetM(A, M) ;
    mxSetN(A, N) ;
  } else {
    A = mxCreateNumericMatrix (M, N, classid, mxREAL) ;
  }

  return A ;
}

/** ------------------------------------------------------------------
 ** @brief Create a plain scalar
 **
 ** @param x inital value.
 **
 ** @return new array.
 **/

static mxArray *
uCreateScalar (double x)
{
  mxArray *A = mxCreateDoubleMatrix(1,1,mxREAL) ;
  *mxGetPr(A) = x ;
  return A ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array a numeric scalar?
 **
 ** @param A array to test.
 **
 ** An array is <em>numeric and scalar</em> if:
 ** - It is numeric.
 ** - It as exactly one element.
 **
 ** @return test result.
 **/

static int
uIsScalar(const mxArray* A)
{
  return
    mxIsNumeric (A) && mxGetNumberOfElements(A) == 1 ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array plain matrix?
 **
 ** @param A array to test.
 **
 ** The array @a A satisfies the test if:
 **
 ** - It is a @ref mexutils-plain-matrix "plain matrix"
 ** - @a M < 0 or the number of rows is equal to @a M.
 ** - @a N < 0 or the number of columns is equal to @a N.
 **
 ** @return test result.
 **/

static vl_bool
uIsPlainArray (const mxArray* A)
{
  return
    mxGetClassID(A) == mxDOUBLE_CLASS &&
    ! mxIsComplex(A) &&
    ! mxIsSparse(A) ;
}

static vl_bool
uIsPlainMatrix (const mxArray* A, int M, int N)
{
  return
    uIsPlainArray(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    (M < 0 || mxGetM(A) == M) &&
    (N < 0 || mxGetN(A) == N) ;
}

static vl_bool
uIsPlainVector (const mxArray* A, int M)
{
  return
    uIsPlainArray(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    (mxGetM(A) == 1 || mxGetN(A) == 1) &&
    (M < 0 || (mxGetM(A) == M || mxGetN(A) == M)) ;
}

static vl_bool
uIsPlainScalar (const mxArray* A)
{
  return
    uIsPlainArray(A) &&
    mxGetNumberOfElements(A) == 1 ;
}


/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array a numeric matrix?
 **
 ** @param A array to test.
 ** @param M number of rows.
 ** @param N number of columns.
 **
 ** The array @a A satisfies the test if:
 ** - It is numeric.
 ** - It as two dimensions.
 ** - @a M < 0 or the number of rows is equal to @a M.
 ** - @a N < 0 or the number of columns is equal to @a N.
 **
 ** @return test result.
 **/

static int
uIsMatrix (const mxArray* A, int M, int N)
{
  return
    mxIsNumeric(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    (M < 0 || mxGetM(A) == M) &&
    (N < 0 || mxGetN(A) == N) ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array a vector?
 **
 ** @param A array to test.
 ** @param N number of elements.
 **
 ** The array @a A satisfies the test if
 ** - It is a matrix (see ::uIsMatrix()).
 ** - It has a singleton dimension.
 ** - @c N < 0 or the other dimension is equal to @c N.
 **
 ** @return test result.
 **/

static int
uIsVector(const mxArray* A, int N)
{
  return
    uIsMatrix(A, 1, N) || uIsMatrix(A, N, 1) ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array real?
 **
 ** @param A array to test.
 **
 ** An array satisfies the test if:
 ** - The storage class is DOUBLE.
 ** - There is no imaginary part.
 **
 ** @return test result.
 **/

static int
uIsReal (const mxArray* A)
{
  return
    mxIsDouble(A) &&
    ! mxIsComplex(A) ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array real and scalar?
 **
 ** @param A array to test.
 **
 ** An array is <em>real and scalar</em> if:
 ** - It is real (see ::uIsReal()).
 ** - It as only one element.
 **
 ** @return test result.
 **/

static int
uIsRealScalar(const mxArray* A)
{
  return
    uIsReal (A) && mxGetNumberOfElements(A) == 1 ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array a real matrix?
 **
 ** @param A array to test.
 ** @param M number of rows.
 ** @param N number of columns.
 **
 ** The array @a A satisfies the test if:
 ** - It is real (see ::uIsReal()).
 ** - It as two dimensions.
 ** - @a M < 0 or the number of rows is equal to @a M.
 ** - @a N < 0 or the number of columns is equal to @a N.
 **
 ** @return test result.
 **/

static int
uIsRealMatrix(const mxArray* A, int M, int N)
{
  return
    mxIsDouble(A) &&
    !mxIsComplex(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    (M < 0 || mxGetM(A) == M) &&
    (N < 0 || mxGetN(A) == N) ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array a real vector?
 **
 ** @param A array to test.
 ** @param N number of elements.
 **
 ** The array @a A satisfies the test if
 ** - It is a real matrix (see ::uIsRealMatrix()).
 ** - It has a singleton dimension.
 ** - @c N < 0 or the other dimension is equal to @c N.
 **
 ** @return test result.
 **/

static int
uIsRealVector(const mxArray* A, int N)
{
  return
    uIsRealMatrix(A, 1, N) || uIsRealMatrix(A, N, 1) ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array real with specified dimensions?
 **
 ** @param A array to check.
 ** @param D number of dimensions.
 ** @param dims dimensions.
 **
 ** The array @a A satisfies the test if:
 ** - It is real (see ::uIsReal()).
 ** - @a ndims < 0 or it has @a ndims dimensions and
 **   - for each element of @a dims, either that element is negative
 **     or it is equal to the corresponding dimension of the array.
 **
 ** @return test result.
 **/

static int
uIsRealArray(const mxArray* A, int D, int* dims)
{
  if(!mxIsDouble(A) || mxIsComplex(A))
    return 0 ;

  if(D >= 0) {
    int d ;
    mwSize const * actual_dims = mxGetDimensions(A) ;

    if(mxGetNumberOfDimensions(A) != D)
      return 0 ;

    return 1 ;

    if(dims != NULL) {
      for(d = 0 ; d < D ; ++d) {
        if(dims[d] >= 0 && dims[d] != actual_dims[d])
          return 0 ;
      }
    }
  }
  return 1 ;
}

/** ------------------------------------------------------------------
 ** @deprecated @ref mexutils-array-test
 ** @brief Is the array a string?
 **
 ** @param A array to test.
 ** @param L string length.
 **
 ** The array @a A satisfies the test if:
 ** - its storage class is CHAR;
 ** - it has two dimensions;
 ** - it has one row;
 ** - @a L < 0 or it has @a L columns.
 **
 ** @return test result.
 **/

static int
uIsString(const mxArray* A, int L)
{
  int M = mxGetM(A) ;
  int N = mxGetN(A) ;

  return
    mxIsChar(A) &&
    mxGetNumberOfDimensions(A) == 2 &&
    (M == 1 || (M == 0 && N == 0)) &&
    (L < 0 || N == L) ;
}

/** ------------------------------------------------------------------
 ** @brief Generate MEX error with VLFeat format
 **
 ** @param errorId      error ID string.
 ** @param errorMessage error message C-style format string.
 ** @param ...          format string arguments.
 **
 ** The function invokes @c mxErrMsgTxtAndId.
 **/

void
mxuError(char const * errorId, char const * errorMessage, ...)
{
  char formattedErrorId [512] ;
  char formattedErrorMessage [1024] ;
  va_list args;
  va_start(args, errorMessage) ;

  if (! errorId) {
    errorId = "undefinedError" ;
  }

  if (! errorMessage) {
    errorMessage = "Undefined error description" ;
  }

#ifdef VL_COMPILER_LCC
  sprintf(formattedErrorId,
          "VLFeat:%s", errorId) ;
  vsprintf(formattedErrorMessage,
           errorMessage, args) ;
#else
  snprintf(formattedErrorId,
           sizeof(formattedErrorId)/sizeof(char),
           "VLFeat:%s", errorId) ;
  vsnprintf(formattedErrorMessage,
            sizeof(formattedErrorMessage)/sizeof(char),
            errorMessage, args) ;
#endif
  va_end(args) ;
  mexErrMsgIdAndTxt(formattedErrorId, formattedErrorMessage) ;
}

/** @brief Generate invalid argument error */
#define VLMX_EIA(...) mxuError("invalidArgument", __VA_ARGS__)

/** ------------------------------------------------------------------
 ** @brief Formatted @c mexErrMsgTxt()
 **
 ** @param format format string (for sprintf).
 ** @param ...    format string arguments.
 **/

void
uErrMsgTxt(char const * format, ...)
{
  enum { buffLen = 1024 } ;
  char buffer [buffLen] ;
  va_list args;
  va_start (args, format) ;
#ifdef VL_COMPILER_LCC
  vsprintf(buffer, format, args) ;
#else
  vsnprintf (buffer, buffLen, format, args) ;
#endif
  va_end (args) ;
  mexErrMsgTxt (buffer) ;
}


/** -------------------------------------------------------------------
 ** @brief MEX option
 **/

struct _uMexOption
{
  const char *name ; /**< option name */
  int has_arg ;      /**< has argument? */
  int val ;          /**< value to return */
} ;

/** @brief MEX option type
 ** @see ::_uMexOption
 **/
typedef struct _uMexOption uMexOption ;

/** ------------------------------------------------------------------
 ** @brief Case insensitive string comparison
 **
 ** @param s1 first string.
 ** @param s2 second string.
 **
 ** @return 0 if the strings are equal, >0 if the first string is
 ** greater (in lexicographical order) and <0 otherwise.
 **/

static int
uStrICmp(const char *s1, const char *s2)
{
  while (tolower((unsigned char)*s1) ==
         tolower((unsigned char)*s2))
  {
    if (*s1 == 0)
      return 0;
    s1++;
    s2++;
  }
  return
    (int)tolower((unsigned char)*s1) -
    (int)tolower((unsigned char)*s2) ;
}

/** ------------------------------------------------------------------
 ** @brief Process next option
 **
 ** @param args     MEX argument array.
 ** @param nargs    MEX argument array length.
 ** @param options  List of option definitions.
 ** @param next     Pointer to the next option (in and out).
 ** @param optarg   Pointer to the option optional argument (out).
 **
 ** The function scans the MEX driver arguments array @a args of @a
 ** nargs elements for the next option starting at location @a next.
 **
 ** This argument is supposed to be the name of an option (case
 ** insensitive). The option is looked up in the option table @a
 ** options and decoded as the value uMexOption::val. Furthermore, if
 ** uMexOption::has_arg is true, the next entry in the array @a args
 ** is assumed to be argument of the option and stored in @a
 ** optarg. Finally, @a next is advanced to point to the next option.
 **
 ** @return the code of the option, or -1 if the argument list is
 ** exhausted. In case of an error (e.g. unknown option) the function
 ** prints an error message and quits the MEX file.
 **/

static int uNextOption(mxArray const *args[], int nargs,
                       uMexOption const *options,
                       int *next,
                       mxArray const **optarg)
{
  char err_msg [1024] ;
  char name    [1024] ;
  int opt = -1, i, len ;

  if (*next >= nargs) {
    return opt ;
  }

  /* check the array is a string */
  if (! uIsString (args [*next], -1)) {
    snprintf(err_msg, sizeof(err_msg),
             "The option name is not a string (argument number %d).",
             *next + 1) ;
    mexErrMsgTxt(err_msg) ;
  }

  /* retrieve option name */
  len = mxGetNumberOfElements (args [*next]) ;

  if (mxGetString (args [*next], name, sizeof(name))) {
    snprintf(err_msg, sizeof(err_msg),
             "The option name is too long (argument number %d).",
             *next + 1) ;
    mexErrMsgTxt(err_msg) ;
  }

  /* advance argument list */
  ++ (*next) ;

  /* now lookup the string in the option table */
  for (i = 0 ; options[i].name != 0 ; ++i) {
    if (uStrICmp(name, options[i].name) == 0) {
      opt = options[i].val ;
      break ;
    }
  }

  /* unknown argument */
  if (opt < 0) {
    snprintf(err_msg, sizeof(err_msg),
             "Unknown option '%s'.", name) ;
    mexErrMsgTxt(err_msg) ;
  }

  /* no argument */
  if (! options [i].has_arg) {
    if (optarg) *optarg = 0 ;
    return opt ;
  }

  /* argument */
  if (*next >= nargs) {
    snprintf(err_msg, sizeof(err_msg),
             "Option '%s' requires an argument.", options[i].name) ;
    mexErrMsgTxt(err_msg) ;
  }

  if (optarg) *optarg = args [*next] ;
  ++ (*next) ;
  return opt ;
}

/* MEXUTILS_H */
#endif
