package polyregion.examples

import polyregion.scala.NativeStruct

object Mandelbrot {

  final val Palette = Array[Colour](
    Colour(66, 30, 15),
    Colour(25, 7, 26),
    Colour(9, 1, 47),
    Colour(4, 4, 73),
    Colour(0, 7, 100),
    Colour(12, 44, 138),
    Colour(24, 82, 177),
    Colour(57, 125, 209),
    Colour(134, 181, 229),
    Colour(211, 236, 248),
    Colour(241, 233, 191),
    Colour(248, 201, 95),
    Colour(255, 170, 0),
    Colour(204, 128, 0),
    Colour(153, 87, 0),
    Colour(106, 52, 3)
  )

  case class Complex(real: Double, imag: Double) {
    def inverse: Complex = {
      val denom = real * real + imag * imag
      Complex(real / denom, -imag / denom)
    }
    def +(b: Complex): Complex    = Complex(real + b.real, imag + b.imag)
    def -(b: Complex): Complex    = Complex(real - b.real, imag - b.imag)
    def *(b: Complex): Complex    = Complex(real * b.real - imag * b.imag, real * b.imag + imag * b.real)
    def /(b: Complex): Complex    = this * b.inverse
    def unary_- : Complex         = Complex(-real, -imag)
    def abs: Double               = math.hypot(real, imag)
    override def toString: String = s"$real + ${imag}i"
  }

  def interpolate(input: Double, inputMin: Double, inputMax: Double, outputMin: Double, outputMax: Double): Double =
    ((outputMax - outputMin) * (input - inputMin) / (inputMax - inputMin)) + outputMin

  case class Colour(packed: Int) extends AnyVal {
    def a: Int = (packed >> 24) & 0xff
    def r: Int = (packed >> 16) & 0xff
    def g: Int = (packed >> 8) & 0xff
    def b: Int = packed & 0xff
    def mix(that: Colour, x: Double): Colour = Colour(
      r = ((r - that.r) * x + that.r).toInt,
      g = ((g - that.g) * x + that.g).toInt,
      b = ((b - that.b) * x + that.b).toInt,
      a = ((a - that.a) * x + that.a).toInt
    )
  }

  object Colour {
    final val Black: Colour             = Colour(0, 0, 0)
    final val White: Colour             = Colour(255, 255, 255)
    private def clampUInt8(x: Int): Int = x.max(0).min(255)
    def apply(r: Int, g: Int, b: Int, a: Int = 255): Colour = Colour(
      (clampUInt8(a) & 0xff) << 24     //
        | (clampUInt8(r) & 0xff) << 16 //
        | (clampUInt8(g) & 0xff) << 8  //
        | (clampUInt8(b) & 0xff)       //
    )
  }

  def itMandel(c: Complex, imax: Int, bailout: Int): (Complex, Int) = {
    var z = Complex(0d, 0d)
    var i = 0
    while (z.abs <= bailout && i < imax) {
      z = z * z + c
      i += 1
    }
    (z, i)
  }

  case class ItResult(c: Complex, i: Int)

  def itMandel2(c: Complex, imax: Int, bailout: Int): ItResult = {
    var z = Complex(0d, 0d)
    var i = 0
    while (z.abs <= bailout && i < imax) {
      z = z * z + c
      i += 1
    }
    ItResult(z, i)
  }

  def mkColour(z: Complex, iter: Int, maxIter: Int): Colour =
    if (iter >= maxIter) Colour(0, 0, 0) // Colour.Black
    else {
      val logZn = math.log(z.abs) / 2
      val nu    = math.log(logZn / math.log(2)) / math.log(2)
      Colour(0, 0, 0)
      // Palette(iter % Palette.length).mix(Palette((iter + 1) % Palette.length), nu)
    }

  // given NativeStruct[Complex] = polyregion.compiletime.nativeStructOf
  // given NativeStruct[ItResult] = polyregion.compiletime.nativeStructOf

  def run(buffer: Array[Array[Colour]], width: Int, height: Int, poiX: Double, poiY: Double, scale: Double): Unit = {
    val maxIter      = 500
    val zoom         = 1d / scale
    val (xMin, xMax) = (poiX - zoom, poiX + zoom)
    val (yMin, yMax) = (poiY - zoom, poiY + zoom)
    import scala.collection.parallel.CollectionConverters.*
    for {
      y <- (0 until height)
      x <- 0 until width
    } {

      val m2 = polyregion.scala.compiletime.offload {
        val c = Complex(interpolate(x, 0, width, xMin, xMax), interpolate(y, 0, height, yMin, yMax))
        val t = itMandel2(c, maxIter, 4)
//        val m = mkColour(t.c, t.i, maxIter)
        ()
      }

      // val c         =   polyregion.scala.compiletime.offload { Complex(interpolate(x, 0, width, xMin, xMax), interpolate(y, 0, height, yMin, yMax))  }
      // val (z, iter) = itMandel(c, maxIter, 4)
      // buffer(x)(y) = mkColour(z, iter, maxIter)
    }
  }

  def showImage(image: java.awt.image.BufferedImage) = {
    import java.awt.Color
    import java.awt.BorderLayout

    import javax.swing.*

    val frame: JFrame = new JFrame
    frame.setTitle("Mandelbrot")

    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    frame.getContentPane.setBackground(Color.BLACK)
    frame.getContentPane.setLayout(new BorderLayout)
    frame.getContentPane.add(new JLabel(new ImageIcon(image)))
    frame.pack()
    frame.setVisible(true)
  }

  def main(args: Array[String]): Unit = {
    import java.awt.image.BufferedImage

    println("Go")
    val image = new BufferedImage(1024, 1024, BufferedImage.TYPE_INT_ARGB)

    val data = Array.ofDim[Colour](image.getWidth, image.getHeight)

    val N       = 10
    val elapsed = Array.ofDim[Long](N)
    for (i <- 0 until N) {
      val start = System.nanoTime()
//      run(data, image.getWidth, image.getHeight, 0.28693186889504513, 0.014286693904085048, 10000 + i)
      run(data, image.getWidth, image.getHeight, -0.7, 0, 1.0)
      elapsed(i) = System.nanoTime() - start
    }

    println(elapsed.map(_.toDouble / 1e6).mkString(", "))

    for {
      x <- 0 until image.getWidth
      y <- 0 until image.getHeight
    } yield image.setRGB(x, y, data(x)(y).packed)
    println("Done")

    showImage(image)

  }

}
