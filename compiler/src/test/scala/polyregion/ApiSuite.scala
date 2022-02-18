package polyregion

import polyregion.compiletime.*

import scala.compiletime.*

class ApiSuite extends BaseSuite {

  test("reduction") {
    if (Toggles.ApiSuite) {
//      reduce(0 to 100)(0d, i => i.toDouble)(_ * _)
    }
  }

}
