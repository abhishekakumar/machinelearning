import matplotlib.pyplot as plt
import pylab


def plot(values, scores, xlabel, ylabel, title, xscale):
    print '\nPlotting Graph. Please close the graph to continue'
    svm_fig1 = plt.figure(figsize=(8, 6), dpi=80)
    svm1 = svm_fig1.add_subplot(111)
    svm1.plot(values, scores)
    svm1.set_xlabel(xlabel + ' ( %s )' % xscale)
    svm1.set_ylabel(ylabel)
    svm1.set_xscale(xscale)
    svm1.set_title(title, fontsize=12)
