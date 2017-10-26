package com.minosys.app.ml.lib;

public interface ILossFunction {
	float derivative(float y, float in);
}
