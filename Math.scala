package utils 

import breeze.linalg._
import breeze.stats._
import breeze.numerics._

object Math{

	def readCSV(path: String) = {
		import java.io._
		breeze.linalg.csvread(new File(path))
	}

	def readMAT(path:String, varName: String):DenseMatrix[Double] = {
		// read MAT file
		import com.jmatio.io._
		import java.io._
		val a = new MatFileReader(new File(path))
		val mla : com.jmatio.types.MLDouble = a.getMLArray(varName).asInstanceOf[com.jmatio.types.MLDouble]
		val mlaA = mla.getArray()

		var out = DenseMatrix.zeros[Double](mla.getM(),mla.getN())

		var i:Int=0
		while(i<mla.getM()){
			var j:Int=0
			while(j<mla.getN()){
				out(i,j) = mlaA(i)(j)

				j += 1
			}
			i += 1
		}

		out
   }

	object LinAlg{

		def sigmoidGradient(X:breeze.linalg.DenseMatrix[Double]):breeze.linalg.DenseMatrix[Double] = {
			sigmoid(X):*(sigmoid(X):*(-1.0):+1.0)
		}

		def sigmoidGradient(X:breeze.linalg.DenseVector[Double]):breeze.linalg.DenseVector[Double] = {
			sigmoid(X):*(sigmoid(X):*(-1.0):+1.0)
		}

		// add one column of ones in front of matrix - used before inputting into regressions
		def addOnes(X:breeze.linalg.DenseMatrix[Double]):breeze.linalg.DenseMatrix[Double] = DenseMatrix.horzcat(DenseMatrix.ones[Double](X.rows, 1),X)


	}


}