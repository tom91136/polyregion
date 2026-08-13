package polyregion.spectra

object check_surface {
  def compile(api: SpectraApi, input: Array[Int], output: Array[Int]): Unit = {
    val floats  = new Array[Float](input.length)
    val doubles = new Array[Double](input.length)
    api.transform(input, floats, input.length, _.toFloat + 1)
    api.transform_binary(input, floats, doubles, input.length, _.toDouble + _)
    api.reduce(input, input.length, 0, _ + _)
    api.exclusive_scan(input, output, input.length, 0, _ + _)
    api.transform_inclusive_scan(input, doubles, input.length, _.toDouble, _ + _)
    api.transform_exclusive_scan(input, doubles, input.length, 0.0, _.toDouble, _ + _)
    api.adjacent_difference(input, output, input.length, _ - _)
    api.for_each(output, output.length, _ + 1)
    api.generate(output, output.length, () => 1)
    api.copy(input, input.length, output)
    api.all_of(input, input.length, _ > 0)
    api.count(input, input.length, 1)
    api.inner_product(input, floats, input.length, 0.0, _ + _, _.toDouble * _)
    api.transform_reduce(input, input.length, 0.0, _.toDouble + 1, _ + _)
    api.set_intersection(input, input.length, input, input.length, output, output.length, _ < _)
    api.sort_by_key(input, floats, input.length, _ < _)
    api.reduce_by_key(input, floats, output, floats, input.length, _ == _, _ + _)
  }
}
