import contextlib
import joblib
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from joblib import Parallel, delayed

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    # from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/49950707
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def pmap(f, arr, n_jobs=-1, notebook=False, **kwargs):
    _tqdm = tqdm_nb if notebook else tqdm
    arr = list(arr)
    with tqdm_joblib(_tqdm(total=len(arr))) as progress_bar:
        return Parallel(n_jobs=n_jobs, prefer='processes')(delayed(f)(i) for i in arr)
