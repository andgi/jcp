// Copyright (C) 2015  Anders Gidenstam
// License: to be defined.
package jcp.util;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

/**
 * Base class for parallel actions over contiguous int intervals.
 *
 * @author anders.gidenstam(at)hb.se
 */
public abstract class ParallelizedAction
    extends RecursiveAction
{
    private static final ForkJoinPool taskPool = new ForkJoinPool();

    protected int _first;
    protected int _last;

    /**
     * Constructs a set of actions for the interval [first, last).
     *
     * @param first the first index
     * @param last  the index after the last index in the interval.
     */
    public ParallelizedAction(int first, int last)
    {
        _first = first;
        _last = last;
    }

    /**
     * Starts this set of actions.
     */
    public void start()
    {
        taskPool.invoke(this);
    }

    /**
     * The action to be performed for index i.
     *
     * @param i the index
     */
    protected abstract void compute(int i);

    /**
     * Creates a ParallelizedAction for the sub-interval.
     *
     * @param first the first index in the sub-interval
     * @param the the index after the last index in the sub-interval.
     */
    protected abstract ParallelizedAction createSubtask(int first, int last);

    /**
     * Inherited from RecursiveAction. Do not overrride.
     */
    protected final void compute()
    {
        if (_last - _first < 100) {
            for (int i = _first; i < _last; i++) {
                compute(i);
            }
        } else {
            int split = (_last - _first)/2;
            invokeAll(createSubtask(_first, _first + split),
                      createSubtask(_first + split, _last));
        }
    }
}
