package polyregion

object check_surface {
  def compile(input: Array[Int], output: Array[Int]): Unit = InterfaceSurfaceCheck.erase {
    val floats  = new Array[Float](input.length)
    val doubles = new Array[Double](input.length)
    spectra.transform(input, floats, input.length, _.toFloat + 1)
    spectra.transform_binary(input, floats, doubles, input.length, _.toDouble + _)
    spectra.reduce(input, input.length, 0, _ + _)
    spectra.exclusive_scan(input, output, input.length, 0, _ + _)
    spectra.transform_inclusive_scan(input, doubles, input.length, _.toDouble, _ + _)
    spectra.transform_exclusive_scan(input, doubles, input.length, 0.0, _.toDouble, _ + _)
    spectra.adjacent_difference(input, output, input.length, _ - _)
    spectra.for_each(output, output.length, _ + 1)
    spectra.generate(output, output.length, () => 1)
    spectra.copy(input, input.length, output)
    spectra.all_of(input, input.length, _ > 0)
    spectra.count(input, input.length, 1)
    spectra.inner_product(input, floats, input.length, 0.0, _ + _, _.toDouble * _)
    spectra.transform_reduce(input, input.length, 0.0, _.toDouble + 1, _ + _)
    spectra.set_intersection(input, input.length, input, input.length, output, output.length, _ < _)
    spectra.sort_by_key(input, floats, input.length, _ < _)
    spectra.reduce_by_key(input, floats, output, floats, input.length, _ == _, _ + _)
  }
}
