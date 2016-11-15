// JCP - Java Conformal Prediction framework
// Copyright (C) 2016  Anders Gidenstam
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

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * A <tt>FIFOParallelExecutor</tt> executes <tt>Callable</tt>s in parallel
 * and returns the results in the order the <tt>Callable</tt>s were issued.
 *
 * The class does not implement the Java <tt>Executor</tt> interface despite
 * its name.
 *
 * @author anders.gidenstam(at)hb.se
 * @param <E> the type of the result returned by the <tt>Callable</tt>s.
 */
public class FIFOParallelExecutor<E>
{
    private ExecutorService _threadPool;
    private BlockingQueue<Future<E>> _queue;

    /**
     * Creates a <tt>FIFOParallelExecutor</tt> using the provided
     * <tt>ExecutorService</tt> to execute the <tt>Callable</tt>s.
     * @param executorService the <tt>ExecutorService</tt> used to execute the <tt>Callable</tt>s.
     */
    public FIFOParallelExecutor(ExecutorService executorService)
    {
        this(Runtime.getRuntime().availableProcessors(), executorService);
    }

    /**
     * Creates a <tt>FIFOParallelExecutor</tt> using the provided
     * level of parallelism and the <tt>ExecutorService</tt> to
     * execute the <tt>Callable</tt>s.
     * @param parallelism      the level of parallelism to use.
     * @param executorService  the <tt>ExecutorService</tt> used to execute the <tt>Callable</tt>s.
     */
    public FIFOParallelExecutor(int parallelism,
                                ExecutorService executorService)
    {
        this(new ArrayBlockingQueue<Future<E>>(parallelism), executorService);
    }

    /**
     * Creates a <tt>FIFOParallelExecutor</tt> using the provided
     * <tt>BlockingQueue</tt> and the <tt>ExecutorService</tt> to
     * execute the <tt>Callable</tt>s.
     * @param queue           the blocking queue used to store the submitted <tt>Callable</tt>s.
     * @param executorService the <tt>ExecutorService</tt> used to execute the <tt>Callable</tt>s.
     */
    public FIFOParallelExecutor(BlockingQueue<Future<E>> queue,
                                ExecutorService executorService)
    {
        _queue = queue;
        _threadPool = executorService;
    }

    /**
     * Submits the <tt>Callable</tt> for execution. Blocks if the level of
     * parallelism is exceeded.
     *
     * @param e  a <tt>Callable&lt;E&gt;</tt> that is to be executed.
     * @throws java.lang.InterruptedException
     */
    public void submit(Callable<E> e) throws InterruptedException
    {
        // FIFO order is guaranteed if there is a single enqueuer.
        Future<E> futureResult = _threadPool.submit(e);
        _queue.put(futureResult);
    }

    /**
     * Returns the next result, blocking to await its arrival or completion if
     * needed.
     *
     * @return the next result from the submitted <tt>Callable</tt>s.
     * @throws java.util.concurrent.ExecutionException
     * @throws java.lang.InterruptedException
     */
    public E take() throws ExecutionException, InterruptedException
    {
        // FIFO order of results is guaranteed if there is a single dequeuer.
        Future<E> e = _queue.take();
        return e.get();
    }
}
