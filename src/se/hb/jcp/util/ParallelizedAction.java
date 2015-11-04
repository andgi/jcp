// JCP - Java Conformal Prediction framework
// Copyright (C) 2015  Anders Gidenstam
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
package se.hb.jcp.util;

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
    private static final int MIN_WORK = 1;
    private static final int MAX_DEPTH = 6;
    private static final ForkJoinPool taskPool = new ForkJoinPool();

    private int _first;
    private int _last;
    private int _depth;

    /**
     * Constructs a set of actions for the interval [first, last).
     *
     * @param first  the first index
     * @param last   the index after the last index in the interval
     */
    public ParallelizedAction(int first, int last)
    {
        _first = first;
        _last  = last;
        _depth = 0;
    }

    /**
     * Starts this set of actions.
     */
    public void start()
    {
        taskPool.invoke(this);
    }

    /**
     * Performs any needed intialization for the sub-interval once the split
     * threshold has been reached. E.g. allocating buffers that can be reused
     * by the sequential compute(i) calls in this task.
     *
     * @param first  the first index in the sub-interval
     * @param last   the index after the last index in the sub-interval
     */
    protected void initialize(int first, int last) {}

    /**
     * Performs any needed finalization for the sub-interval once all
     * sequential compute(i) calls for it has been completed.
     *
     * @param first  the first index in the sub-interval
     * @param last   the index after the last index in the sub-interval
     */
    protected void finalize(int first, int last) {}

    /**
     * The action to be performed for index i.
     *
     * @param i  the index in the interval
     */
    protected abstract void compute(int i);

    /**
     * Creates a ParallelizedAction for the sub-interval.
     *
     * @param first  the first index in the sub-interval
     * @param last   the index after the last index in the sub-interval
     */
    protected abstract ParallelizedAction createSubtask(int first, int last);

    /**
     * Creates a ParallelizedAction for the sub-interval.
     *
     * @param first  the first index in the sub-interval
     * @param last   the index after the last index in the sub-interval
     * @param depth  the current depth of task subdivision.
     */
    private ParallelizedAction createSubtask(int first, int last, int depth)
    {
        ParallelizedAction a = createSubtask(first, last);
        _depth = depth;
        return a;
    }

    /**
     * Inherited from RecursiveAction. Do not overrride.
     */
    protected final void compute()
    {
        if ((_depth >= MAX_DEPTH) || (_last - _first <= MIN_WORK)) {
            initialize(_first, _last);
            for (int i = _first; i < _last; i++) {
                compute(i);
            }
            finalize(_first, _last);
        } else {
            int split = (_last - _first)/2;
            invokeAll(createSubtask(_first, _first + split, _depth + 1),
                      createSubtask(_first + split, _last, _depth + 1));
        }
    }
}
