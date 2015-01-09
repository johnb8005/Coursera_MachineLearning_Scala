package models

import breeze.linalg._
import breeze.math._ 		// needed for universal functions such as exp
import breeze.numerics._
import breeze.optimize._

import scala.collection._

/*
	@arg X: input data (the ones column is not added)
	@arg y: output data
	@arg layers: describes the structruee of the hidden layers, input and output layers are defined by data
*/
class NeuralNetwork(layers: Seq[Int], x_in: DenseMatrix[Double], y_in: DenseMatrix[Double], alpha_in: Double, epochs_in: Int){
	

	/*
		prepare and store data
	*/
	// training data
	val x:DenseMatrix[Double] 				= x_in
	val y:DenseMatrix[Double] 				= y_in


	/*
		computes dimensions
	*/
	// number of observations
	val m:Int 									= {if(x.rows==y.rows){x.rows}else{0}}
	// number of input paramters
	val n:Int 									= x.cols
	// number of output param
	val p:Int 									= y.cols

	/*
		initialize layer
	*/
	// layers structure
	val s:Seq[Int] 							= Seq(n) ++ layers :+p

	println(s)

	// number of layers
	val L:Int 									= s.size

	/*
		optimization parameters
	*/
	val alpha:Double 							= alpha_in
	val epochs:Int 							= epochs_in

	/*
		init matrices
	*/
	var a:mutable.Buffer[DenseMatrix[Double]] 		= {
		var a:mutable.Buffer[DenseMatrix[Double]] = mutable.Buffer()
		s.map{sl =>
			a = a :+ DenseMatrix.zeros[Double](m, sl)
		}
		a
	}

	var d:mutable.Buffer[DenseMatrix[Double]] 		= {
		var d:mutable.Buffer[DenseMatrix[Double]] = mutable.Buffer()
		s.map{sl =>
			d = d:+ DenseMatrix.zeros[Double](m, sl)
		}
		d
	}

	var aP:mutable.Buffer[DenseMatrix[Double]] 		= {
		var aP:mutable.Buffer[DenseMatrix[Double]] = mutable.Buffer()
		s.map{sl =>
			aP = aP :+ DenseMatrix.zeros[Double](m, sl+1)
		}
		aP
	}

	var dP:mutable.Buffer[DenseMatrix[Double]] = {
		var dP:mutable.Buffer[DenseMatrix[Double]] = mutable.Buffer()
		s.map{sl =>
			dP = dP :+ DenseMatrix.zeros[Double](m, sl+1)
		}
		dP
	}

	var theta:mutable.Buffer[DenseMatrix[Double]]		= {
		var theta:mutable.Buffer[DenseMatrix[Double]] = mutable.Buffer()
		var sk = 0
		s.map{sl =>
			if(sk>0){
				theta = theta :+ {(DenseMatrix.rand(sk, sl):* 2.0 :-1.0):/4.0}
			}
			sk = sl
		}
		theta
	}

	import play.Logger
	Logger.info(theta.toString)
	//Logger.info(theta(0).toString)
	

	var thetaP:mutable.Buffer[DenseMatrix[Double]]		= {
		var thetaP:mutable.Buffer[DenseMatrix[Double]] = mutable.Buffer()
		var sk = 0
		s.map{sl =>
			if(sk>0){
				thetaP = thetaP :+ DenseMatrix.zeros[Double](sk+1, sl)
			}
			sk 	= sl
		}
		thetaP
	}


	def thetaSize{
		thetaP.map{t=>
			println(t.rows.toString+" "+t.cols.toString)
		}
		
	}
	/*
		end: init matrices
	*/

	//thetaSize
	var k = 0
	while(k < epochs){
		//println("== forwardPropagation ==")
		forwardPropagation
		//println("== backwardPropagation ==")
		backwardPropagation
		gradient

		k += 1
	}

	//thetaSize
	//println(aP)
	//println(thetaP)


	def forwardPropagation{

		//println("theta")
		//println(thetaP)

		a(0) = x
		aP(0) = 
			DenseMatrix.horzcat(
				DenseMatrix.ones[Double](m,1)
				,a(0)
			)

		//println(aP(0))
		//println(dP(0))

		//println(aP(0).cols)
		//println(aP(0).rows)

		var i = 0
		for (i <- 1 to L-1){
			//println(i)
			a(i) = sigmoid(aP(i-1)*thetaP(i-1))
			//println(a(i))
			aP(i) = 
			DenseMatrix.horzcat(
				DenseMatrix.ones[Double](m,1)
				,a(i)
			)

			//println("size")
			//println(aP(i).cols)
			//println(aP(i).rows)

			//println("size theta")
			//println(thetaP(i-1).cols)
			//println(thetaP(i-1).rows)
		} 
	}

	def backwardPropagation{

		d(L-1) = a(L-1) - y

		var j = 0
		var i = 0
		for (j <- 0 to L-2){
			i = L-2 - j
			//println("here"+i.toString)
			d(i) = d(i+1)*theta(i).t :* sigmoid(a(i)) :* (-sigmoid(a(i)):+1.0)
		}

	}

	def gradient{
		var i = 0
		for(i <- 0 to L-2){
			thetaP(i) += aP(i).t*d(i+1) :* alpha/m
		}
	}
}