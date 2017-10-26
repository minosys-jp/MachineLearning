package com.minosys.app.ml.lib;

public class QuadLossFunction implements ILossFunction {

	@Override
	public float derivative(float y, float in) {
		// TODO 自動生成されたメソッド・スタブ
		return in - y;
	}

}
