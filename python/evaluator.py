import os
import time
from multiprocessing import Pool
from config import config as c

def eval_onetime(w_no, best_no, result_dir):
    os.system("../c/build/gomokuer.exe weval -w0 {wdir}/{w}.pkl -w1 {wdir}/{best}.pkl -r {r} -o {o} -v 0".
            format(
                wdir=c.weights_dir,
                w=w_no,
                best=best_no,
                r=1,
                o=result_dir))

def run():
    print("Evaluating...")
    queue = os.listdir(c.weights_dir + "/eval/queue")
    queue_int = []

    for w in queue:
        queue_int.append(int(w))

    queue_int.sort()

    for w in queue_int:
        best = os.listdir(c.weights_dir + "/best")[0]

        best_no = int(best)
        w_no = int(w)
        if w_no <= best_no:
            continue
        print("weight {w} vs {best}".format(w=w_no, best=best_no))
        result_dir = c.weights_dir + "/eval/result/{0}_{1}".format(w_no, best_no)

        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        p = Pool(c.eval_process_count)
        for i in range(c.eval_target_rounds):
            p.apply_async(eval_onetime, args = (w, best, result_dir))
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
        logstr = "{0}_{1} WLRatio {2:.3} ".format(w_no, best_no, ratio)
        print("weight {w} vs {best} result:".format(
            w=w_no, best=best_no))
        print("win={win}, draw={draw}, loss={loss}".format(
            win=win, draw=draw, loss=loss))
        print("win/loss={ratio}".format(ratio=ratio))
        if ratio >= 1.2:
            print("weight {w} become the best weight!".format(w=w_no))
            os.rename(c.weights_dir + "/best/{0}".format(best_no),
                      c.weights_dir + "/best/{0}".format(w_no))
            logstr += "PASS"
        else:
            print("weight {w} failed.".format(w=w_no))
            logstr += "FAIL"
        os.remove(c.weights_dir + "/eval/queue/{0}".format(w_no))

        with open(c.weights_dir + "/log/" + logstr, mode="w"):
            pass
    print("Evaluate Finished!")


if __name__ == "__main__":
    run()