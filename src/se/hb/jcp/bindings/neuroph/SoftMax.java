// JCP - Java Conformal Prediction framework
// Copyright (C) 2024  Tom le Cam
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
// The public interface is based on cern.colt.matrix.DoubleMatrix2D.
package se.hb.jcp.bindings.neuroph;

import org.neuroph.core.Layer;
import org.neuroph.core.Neuron;
import org.neuroph.core.transfer.TransferFunction;

/**
 * Activation function which enforces that output neurons have probability distribution (sum of all outputs is one)
 */
public class SoftMax extends TransferFunction {

    private Layer layer;
    private double totalLayerInput;

    public SoftMax(final Layer layer) {
        this.layer = layer;
    }


    @Override
    public double getOutput(double netInput) {
        totalLayerInput = 0;
        double max = 0;
        
        for (Neuron neuron : layer.getNeurons()) {
            totalLayerInput += Math.exp(neuron.getNetInput()-max);
        }

        output = Math.exp(netInput-max) / totalLayerInput;
        return output;
    }

    @Override
    public double getDerivative(double net) {
        return output * (1d - output); 
  
    }
}
