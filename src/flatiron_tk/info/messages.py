import fenics as fe

def info(msg, all_rank=False):
    if all_rank:
        print(msg)
    else:
        if fe.MPI.comm_world.rank == 0:
            print(msg)

def warning(msg, all_rank=False):
    wrn_msg = 'WARNING: ' + msg
    info(wrn_msg, all_rank)

def error(err_type, err_msg):
    raise err_type(err_msg)
