package com.minosys.app.ml.lib;

/**
 * Training Neural Net
 * @author minoru
 *
 */
public interface INNTrainer {
	public void setTeacher(ImageLabelSet ils, INoiseShaper shaper);
	public void setLabels(ImageLabelSet lastIls);
	public void train(NeuralNet nn);
}
