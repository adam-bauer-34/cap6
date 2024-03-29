"""Tools file for CAP6. 

Adam Michael Bauer
University of Illinois at Urbana Champaing
adammb4@illinois.edu
1..18.2023

A set of tools used throughout CAP6. Many of these were originally
written by the EZClimate team.
"""

import csv
import io

import numpy as np
from numba import jit

###########
### I/O ###
###########

def find_path(file_name, directory="data", file_type=".csv"):
    import os
    cwd = os.getcwd()
    if not os.path.exists(directory):
        os.makedirs(directory)
    d = os.path.join(cwd, os.path.join(directory,file_name+file_type))
    return d

def create_file(file_name):
    import os
    d = find_path(file_name)
    if not os.path.isfile(d):
        open(d, 'w').close()
    return d

def file_exists(file_name):
    import os
    d = find_path(file_name)
    return os.path.isfile(d)

def load_csv(file_name, delimiter=';', comment=None):
    d = find_path(file_name)
    pass

def clean_lines(f):
    """
    Filter out blank lines to avoid prior cross-platform line termination problems.
    """
    lines = f.read().splitlines()
    lines = [line for line in lines if line.strip()]
    content = '\n'.join(lines)
    sio = io.StringIO()
    sio.write(content)
    sio.seek(0)
    return sio

def write_columns_csv(lst, file_name, header=[], index=None, start_char=None, delimiter=';', open_as='w'):
    """
    write_columns_csv outputs tree data to an NEW (not existing) csv file

    lst       : a list of a list containing data for a single tree
    file_name :
    headers   : names of the trees; these are put in the first row of the csv file.
    index     : index data (e.g., Year and Node)
     """
    d = find_path(file_name)
    if index is not None:
        index.extend(lst)
        output_lst = list(zip(*index))
    else:
        output_lst = list(zip(*lst))

    with open(d, open_as) as f:
        writer = csv.writer(f, delimiter=delimiter)
        if start_char is not None:
            writer.writerow([start_char])
        if header:
            writer.writerow(header)
        for row in output_lst:
            writer.writerow(row)

def clean_lines(f):
    """
    Filter out blank lines in the given file in order to avoid
    cross-platform line termination problems that
    previously led to data files with blank lines.
    """
    lines = f.read().splitlines()
    lines = [line for line in lines if line.strip()]
    content = '\n'.join(lines)
    sio = io.StringIO()
    sio.write(content)
    sio.seek(0)
    return sio


def write_columns_to_existing(lst, file_name, header="", delimiter=';'):
    """
    writes the tree elements in lst to and EXISTING file with name file_name.
    """
    is_nested_list = lst and (isinstance(lst[0], list) or
                                isinstance(lst[0], np.ndarray))
    if is_nested_list:
        lst = list(zip(*lst))   # transpose columns -> rows

    file_path = find_path(file_name)
    output_rows = []

    # read and extend input
    with open(file_path, 'r') as finput:
        reader = csv.reader(clean_lines(finput), delimiter=delimiter)

        # extend header row
        row = next(reader)
        row.extend(header if is_nested_list else [header])
        output_rows.append(row)

        # extend rest of the rows
        for i,row in enumerate(reader):
            row.extend(lst[i] if is_nested_list else [lst[i]])
            output_rows.append(row)

    # emit output, overwriting original file
    with open(file_path, 'w') as foutput:
        writer = csv.writer(foutput, delimiter=delimiter)
        writer.writerows(output_rows)

def append_to_existing(lst, file_name, header="", index=None, delimiter=';', start_char=None):
    write_columns_csv(lst, file_name, header, index, start_char=start_char, delimiter=delimiter, open_as='a')

def import_csv(file_name, delimiter=';', header=True, indices=None, start_at=0, break_at='\n', ignore=""):
    d = find_path(file_name)
    input_lst = []
    indices_lst = []
    with open(d, 'r') as f:
        reader = csv.reader(clean_lines(f), delimiter=delimiter)
        for _ in range(0, start_at):
            next(reader)
        if header:
            header_row = next(reader)
        for row in reader:
            if row[0] == break_at:
                break
            if row[0] == ignore:
                continue
            if indices:
                input_lst.append(row[indices:])
                indices_lst.append(row[:indices])
            else:
                input_lst.append(row)
    if header and not indices :
        return header_row, np.array(input_lst, dtype="float64")
    elif header and indices:
        return header_row[indices:], indices_lst, np.array(input_lst, dtype="float64")
    return np.array(input_lst, dtype="float64")


##########
### MP ###
##########

def _pickle_method(method):
    func_name = method.__func__.__name__
    obj = method.__self__
    cls = method.__self__.__class__
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


# --------------------------------------
# MATH FUNCTIONS
# --------------------------------------

@jit(nopython=True)
def get_integration(integrand, xvals):
    """Compute an integral.
    
    Compute the definite integral over the assigned x values.
    
    Parameters
    ---------
    integrand: nd array
        integrand values
    xvals: nd array
        x values at which the integrand is evaluated
    
    Returns
    -------
    x: float
        the definite integral over the range of xvals
    """
    
    x = 0
    
    for i in range(1, len(xvals)):
        x += 0.5 * (float(xvals[i]) - float(xvals[i-1])) * (float(integrand[i]) + float(integrand[i-1]))
    
    return x

@jit(nopython=True)
def get_integral_var_ub(integrand, integrand_x_vals, new_xs):
    """Calculate the integral with variable upper bound.
    
    This function evaluates the function:
    f(x) = \int_{x_0}^{x} g(z) dz
    where x_0 is the first x value in integrand_x_vals.
    
    Parameters
    ----------
    integrand: nd array
        the integrand, evaluated at x values integrand_x_vals
    integrand_x_vals: nd array
        x values that the integrand is evaluated at 
    new_x: nd array
        x values we want the new function to be evaluated at
    
    Returns
    -------
    f: nd array
        the result
    """
    
    # combine x values, take out the duplicates (using np.unique),
    # and interpolate to the all the values
    both_x = np.hstack((integrand_x_vals, new_xs))
    no_dups_full_x = np.unique(both_x)
    new_integrand = np.interp(no_dups_full_x, integrand_x_vals, integrand)
    
    # make f values
    f = np.zeros_like(new_xs, dtype=np.float32)
    i = 0
    
    # eval at new upper bounds
    for new_x in new_xs:
        tmp_xs = no_dups_full_x[no_dups_full_x <= new_x]
        f[i] = get_integration(new_integrand, tmp_xs)
        i += 1
        
    return f
