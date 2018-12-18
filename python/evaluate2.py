import os
import time
import random
from multiprocessing import Pool
from config import config as c

def eval_onetime(w_no, best_no, result_dir):
    os.system("../c/build/FastGomoku weval -w0 {wdir}/{w}.pkl -w1 {wdir}/{best}.pkl -s {s} -o {o} -v 1 -k 48000".
            format(
                wdir=c.weights_dir,
                w=w_no,
                best=best_no,
                s=random.random()*10000,
                o=result_dir))

def run():
    print("Evaluating...")

    best_no = 998
    w_no = 519

    print("weight {w} vs {best}".format(w=w_no, best=best_no))
    result_dir = c.weights_dir + "/eval/result/tmp"

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    p = Pool(c.eval_process_count)
    for i in range(c.eval_target_rounds):
        p.apply_async(eval_onetime, args = (w_no, best_no, result_dir))
    p.close()
    p.join()

    results = os.listdir(result_dir)
    win, draw, loss = 0, 0, 0
    for r in results:
        o = r.split("#")[1]
        if o == "W":
            win += 1
        if o == "D":
            draw += 1
        if o == "L":
            loss += 1
    ratio = win / max(loss, 1)

    print("weight {w} vs {best} result:".format(
        w=w_no, best=best_no))
    print("win={win}, draw={draw}, loss={loss}".format(
        win=win, draw=draw, loss=loss))
    print("win/loss={ratio}".format(ratio=ratio))

    os.system("rm " + c.weights_dir + "/eval/result/tmp -rf")
    # os.remove(c.weights_dir + "/eval/result/tmp")


    print("Evaluate Finished!")


if __name__ == "__main__":
    run()