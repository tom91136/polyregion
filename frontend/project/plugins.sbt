addSbtPlugin("com.timushev.sbt" % "sbt-updates" % "0.6.4")
// addSbtPlugin("org.typelevel"             % "sbt-tpolecat"     % "0.5.2")
addSbtPlugin("com.eed3si9n"       % "sbt-assembly"                  % "2.3.0")
addSbtPlugin("ch.epfl.scala"      % "sbt-scalafix"                  % "0.13.0")
addSbtPlugin("org.scalameta"      % "sbt-scalafmt"                  % "2.5.2")
addSbtPlugin("com.lightbend.sbt"  % "sbt-java-formatter"            % "0.8.0")
addSbtPlugin("pl.project13.scala" % "sbt-jmh"                       % "0.4.7")
addSbtPlugin("org.scala-js"       % "sbt-scalajs"                   % "1.20.1")
addSbtPlugin("org.scala-native"   % "sbt-scala-native"              % "0.5.11")
addSbtPlugin("org.portable-scala" % "sbt-scalajs-crossproject"      % "1.3.2")
addSbtPlugin("org.portable-scala" % "sbt-scala-native-crossproject" % "1.3.2")

// XXX Scala.js's fullOptJS is dropping gcc, see https://www.scala-js.org/news/2026/04/04/announcing-scalajs-1.21.0
libraryDependencies += "com.google.javascript" % "closure-compiler" % "v20260513"
