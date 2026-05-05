package polyregion.ast

import java.lang.reflect.{Constructor, Field, Method, Modifier}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import scala.collection.immutable.Set
import scala.util.{Failure, Success, Try}

object CppJniBindGen {

  // JNI has specialisations for one reference type: String
  private def isNonJniSpecialisedObjectType(t: Class[?]): Boolean =
    !(t.getName == StringClass || t.isPrimitive || t.isArray)

  private def isVoid(c: Class[?]): Boolean = c == java.lang.Void.TYPE

  private def descriptor(m: Class[?]): String = m match {
    case x if x.equals(java.lang.Double.TYPE)    => "D"
    case x if x.equals(java.lang.Float.TYPE)     => "F"
    case x if x.equals(java.lang.Long.TYPE)      => "J"
    case x if x.equals(java.lang.Integer.TYPE)   => "I"
    case x if x.equals(java.lang.Short.TYPE)     => "S"
    case x if x.equals(java.lang.Character.TYPE) => "C"
    case x if x.equals(java.lang.Boolean.TYPE)   => "Z"
    case x if x.equals(java.lang.Byte.TYPE)      => "B"
    case x if x.equals(java.lang.Void.TYPE)      => "V"
    case x if x.isArray                          => x.getName.replace('.', '/')
    case x                                       => s"L${x.getName.replace('.', '/')};"
  }

  private def descriptor(c: Constructor[?]): String = s"(${c.getParameterTypes.map(descriptor(_)).mkString})"
  private def descriptor(m: Method): String =
    s"(${m.getParameterTypes.map(descriptor(_)).mkString})${descriptor(m.getReturnType)}"
  private def signature(m: Method): String = m.getName + descriptor(m)
  private def descriptorSafe(m: Method): String =
    m.getName + "_" + (m.getParameterTypes.map(descriptor(_)) :+ descriptor(m.getReturnType)).mkString
      .replace('/', '_')
      .replace(';', '_')
      .replace('[', 'a')
      .replace('$', '_')

  private def jniTypeName(t: Class[?]): String = t match {
    case x if x.equals(java.lang.Double.TYPE)                        => "jdouble"
    case x if x.equals(java.lang.Float.TYPE)                         => "jfloat"
    case x if x.equals(java.lang.Long.TYPE)                          => "jlong"
    case x if x.equals(java.lang.Integer.TYPE)                       => "jint"
    case x if x.equals(java.lang.Short.TYPE)                         => "jshort"
    case x if x.equals(java.lang.Character.TYPE)                     => "jcharacter"
    case x if x.equals(java.lang.Boolean.TYPE)                       => "jboolean"
    case x if x.equals(java.lang.Byte.TYPE)                          => "jbyte"
    case x if x.equals(java.lang.Void.TYPE)                          => "void"
    case x if x.isArray && x.getComponentType.getName == StringClass => s"jobjectArray" // jstringArray is not a thing
    case x if x.isArray                                              => s"${jniTypeName(x.getComponentType)}Array"
    case x if x.getName == StringClass                               => "jstring"
    case _                                                           => "jobject"
  }

  private def jniTypedFunctionName(t: Class[?]): String = t match {
    case x if x.equals(java.lang.Double.TYPE)    => "Double"
    case x if x.equals(java.lang.Float.TYPE)     => "Float"
    case x if x.equals(java.lang.Long.TYPE)      => "Long"
    case x if x.equals(java.lang.Integer.TYPE)   => "Int"
    case x if x.equals(java.lang.Short.TYPE)     => "Short"
    case x if x.equals(java.lang.Character.TYPE) => "Char"
    case x if x.equals(java.lang.Boolean.TYPE)   => "Boolean"
    case x if x.equals(java.lang.Byte.TYPE)      => "Byte"
    case x if x.equals(java.lang.Void.TYPE)      => "Void"
    case _                                       => "Object" // even string needs a upcast
  }

  def safeCppNames(name: String): String = name match {
    case x @ ("delete" | "new") => s"${x}_" // TODO add more as they show up
    case x                      => x
  }

  def jniClassName(c: Class[?]): String = c.getName.replace('.', '/')

  private final val StringClass                  = classOf[String].getName
  private final val ObjectClassMethodsSignatures = classOf[AnyRef].getDeclaredMethods.map(m => signature(m)).toSet

  def reflectJniSource(
      knownClasses: Set[String],
      cls: Class[?],
      fp: Field => Boolean,
      cp: Constructor[?] => Boolean,
      mp: Method => Boolean
  ): (String, String) = {

    val publicFields =
      cls.getDeclaredFields
        .filterNot { m =>
          val mod = m.getModifiers
          Modifier.isPrivate(mod) || Modifier.isProtected(mod) || Modifier.isNative(mod) || m.isSynthetic
        }
        .filter(fp)
        .map(f => f.getName -> f.getType)
        .sortBy(_._1)
        .toList

    val publicMethods = cls.getDeclaredMethods
      .filterNot { m =>
        val mod = m.getModifiers
        Modifier.isPrivate(mod) || Modifier.isProtected(mod) || Modifier.isNative(mod) || m.isSynthetic
      }
      .filterNot(m => ObjectClassMethodsSignatures.contains(signature(m)))
      .filter(mp)
      .sortBy(m => descriptorSafe(m))
      .toList

    val ctors = cls.getDeclaredConstructors
      .filter { m =>
        //        val mod = m.getModifiers
        !m.isSynthetic
      }
      .filter(cp)
      .sortBy(_.getParameterCount)
      .toList

    publicFields.map { case (name, tpe) =>
      s"""static jfieldID $name = env->GetFieldID(clazz, "$name", "${descriptor(tpe)}");"""
    }

    val jniName = jniClassName(cls)
    val clsName = cls.getSimpleName

    val metaStructMembers =
      "jclass" -> "clazz"
        :: publicFields.map { case (name, _) => s"jfieldID" -> s"${name}Field" }
        ::: ctors.zipWithIndex.map { case (_, i) => s"jmethodID" -> s"ctor${i}Method" }
        ::: publicMethods.map(f => s"jmethodID" -> s"${descriptorSafe(f)}Method")

    val metaStructMemberInits =
      ("clazz", s"""reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("$jniName")))""", cls.getName)
        :: publicFields.map { case (name, tpe) =>
          (s"${name}Field", s"""env->GetFieldID(clazz, "$name", "${descriptor(tpe)}")""", tpe.getName)
        }
        ::: ctors.zipWithIndex.map { case (ctor, i) =>
          // ctor is (...)V, note the void at the end
          (s"ctor${i}Method", s"""env->GetMethodID(clazz, "<init>", "${descriptor(ctor)}V")""", ctor.toString)
        }
        ::: publicMethods.map { m =>
          val mod = if (Modifier.isStatic(m.getModifiers)) "Static" else ""
          (
            s"${descriptorSafe(m)}Method",
            s"""env->Get${mod}MethodID(clazz, "${m.getName}", "${descriptor(m)}")""",
            m.toString
          )
        }

    val (metaStructStaticMethodFns, metaStructStaticMethodFnImpls) =
      publicMethods
        .filter(m => Modifier.isStatic(m.getModifiers))
        .map { m =>
          val name = safeCppNames(m.getName)
          val args =
            "JNIEnv *env" :: m.getParameters
              .map(p => s"${jniTypeName(p.getType)} ${p.getName}")
              .toList

          val implArgAps =
            ("clazz" :: s"${descriptorSafe(m)}Method" :: m.getParameters.map(_.getName).toList).mkString(", ")

          val proto  = s"${jniTypeName(m.getReturnType)} $name(${args.mkString(", ")}) const;"
          val rtnMod = jniTypedFunctionName(m.getReturnType)
          val rtn    = if (isVoid(m.getReturnType)) "" else "return "
          val impl =
            s"${jniTypeName(m.getReturnType)} $clsName::$name(${args.mkString(", ")}) const { ${rtn}env->CallStatic${rtnMod}Method($implArgAps); }"
          proto -> impl
        }
        .unzip

    val (instStructCtorFactoryFns, instStructCtorFactoryFnImpls) = ctors.zipWithIndex.map { case (ctor, i) =>
      val args = "JNIEnv *env" :: ctor.getParameters.toList.map { p =>
        s"${jniTypeName(p.getType)} ${p.getName}"
      }
      val implArgAps = s"ctor${i}Method" :: ctor.getParameters.map(p => s"${p.getName}").toList

      val proto = s"Instance operator()(${args.mkString(", ")}) const;"
      val impl = s"""$clsName::Instance $clsName::operator()(${args.mkString(", ")}) const {
                    |  return {*this, env->NewObject(clazz, ${implArgAps.mkString(", ")})};
                    |}""".stripMargin

      proto -> impl
    }.unzip

    def delegatedName(tpe: Class[?]) =
      if (!isNonJniSpecialisedObjectType(tpe)) None
      else {
        val ref = tpe
        if (knownClasses.contains(ref.getName)) Some(tpe.getSimpleName)
        else {
          Console.err.println(ref)
          None
        }
      }

    val (instStructFieldGetterFns, instStructFieldGetterFnImpls) =
      publicFields.map { case (field, tpe) =>
        val tpeName = jniTypeName(tpe)
        val getter  = s"env->Get${jniTypedFunctionName(tpe)}Field(instance, meta.${field}Field)"

        delegatedName(tpe) match {
          case Some(cls) =>
            s"$cls::Instance $field(JNIEnv *env, const $cls& clazz) const;" ->
              s"$cls::Instance $clsName::Instance::$field(JNIEnv *env, const $cls& clazz_) const { return {clazz_, $getter}; }"
          case None =>
            val value = tpe match {
              case s if s.getName == StringClass => s"reinterpret_cast<jstring>($getter)"
              case a if a.isArray                => s"reinterpret_cast<$tpeName>($getter)"
              case _                             => getter
            }

            s"$tpeName $field(JNIEnv *env) const;" ->
              s"$tpeName $clsName::Instance::$field(JNIEnv *env) const { return $value; }"
        }
      }.unzip

    val (instStructMethodFns, instStructMethodFnImpls) =
      publicMethods
        .filterNot(m => Modifier.isStatic(m.getModifiers))
        .map { m =>
          val name = safeCppNames(m.getName)
          val tpe  = m.getReturnType
          val args =
            "JNIEnv *env" :: m.getParameters
              .map(p => s"${jniTypeName(p.getType)} ${p.getName}")
              .toList
          val rtnMod = jniTypedFunctionName(tpe)

          val getter = s"env->Call${rtnMod}Method(instance, meta.${descriptorSafe(m)}Method)"

          delegatedName(tpe) match {
            case Some(cls) =>
              val argAndCls = (args :+ s"const $cls& clazz_").mkString(", ")
              s"$cls::Instance $name($argAndCls) const;" ->
                s"$cls::Instance $clsName::Instance::$name($argAndCls) const { return {clazz_, $getter}; }"
            case None =>
              val tpeName = jniTypeName(tpe)

              val value = tpe match {
                case s if s.getName == StringClass => s"reinterpret_cast<jstring>($getter)"
                case a if a.isArray                => s"reinterpret_cast<$tpeName>($getter)"
                case _                             => getter
              }

              val rtn     = if (isVoid(tpe)) "" else "return "
              val justArg = args.mkString(", ")
              s"$tpeName $name($justArg) const;" ->
                s"$tpeName $clsName::Instance::$name($justArg) const { $rtn$value; }"
          }
        }
        .unzip

    val structPrototype = //
      s"""struct $clsName {
         |  struct Instance {
         |    const $clsName &meta;
         |    jobject instance;
         |    Instance(const $clsName &meta, jobject instance);
         |    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
         |    ${(instStructFieldGetterFns ++ instStructMethodFns).mkString("\n    ")}
         |  };
         |${metaStructMembers.map((t, n) => s"  $t $n;").mkString("\n")}
         |private:
         |  explicit $clsName(JNIEnv *env);
         |  static thread_local std::unique_ptr<$clsName> cached;
         |public:
         |  static $clsName& of(JNIEnv *env);
         |  static void drop(JNIEnv *env);
         |  Instance wrap (JNIEnv *env, jobject instance);
         |  ${(metaStructStaticMethodFns ++ instStructCtorFactoryFns).mkString("\n  ")}
         |};""".stripMargin

    val structImpl =
      s"""$clsName::Instance::Instance(const $clsName &meta, jobject instance) : meta(meta), instance(instance) {}
         |${(instStructFieldGetterFnImpls ++ instStructMethodFnImpls).mkString("\n")}
         |$clsName::$clsName(JNIEnv *env)
         |    : ${metaStructMemberInits
          .map { case (m, v, comment) => s"$m($v)" }
          .mkString(s",\n      ")} { };
         |thread_local std::unique_ptr<$clsName> $clsName::cached = {};
         |$clsName& $clsName::of(JNIEnv *env) {
         |  if(!cached) cached = std::unique_ptr<$clsName>(new $clsName(env));
         |  return *cached;
         |}
         |void $clsName::drop(JNIEnv *env){
         |  if(cached) {
         |    env->DeleteGlobalRef(cached->clazz);
         |    cached.reset();
         |  }
         |}
         |$clsName::Instance $clsName::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
         |${(metaStructStaticMethodFnImpls ++ instStructCtorFactoryFnImpls).mkString("\n")}
         |""".stripMargin
    structPrototype -> structImpl
  }

  def generateRegisterNative(cls: Class[?]) = {

    val nativeMethods = cls.getDeclaredMethods
      .filter(m => Modifier.isNative(m.getModifiers))
      .sortBy(m => descriptorSafe(m))

    val registerEntries = nativeMethods.map(m =>
      s"""{(char *)"${m.getName}", (char *)"${descriptor(m)}", (void *)&${safeCppNames(m.getName)}}"""
    )

    val constants = cls.getDeclaredFields
      .filter(f => Modifier.isStatic(f.getModifiers) && Modifier.isFinal(f.getModifiers))
      .filter(_.getType.isPrimitive)
      .sortBy(_.getName)
      .map { f =>
        f.setAccessible(true)
        val rhs = f.getType match {
          case x if x.equals(java.lang.Double.TYPE)    => s"${f.getDouble(null)}"
          case x if x.equals(java.lang.Float.TYPE)     => s"${f.getFloat(null)}"
          case x if x.equals(java.lang.Long.TYPE)      => s"${f.getLong(null)}"
          case x if x.equals(java.lang.Integer.TYPE)   => s"${f.getInt(null)}"
          case x if x.equals(java.lang.Short.TYPE)     => s"${f.getShort(null)}"
          case x if x.equals(java.lang.Character.TYPE) => s"'${f.getChar(null)}'"
          case x if x.equals(java.lang.Boolean.TYPE)   => if (f.getBoolean(null)) "true" else "false"
          case x if x.equals(java.lang.Byte.TYPE)      => s"${f.getByte(null)}"
          case _                                       => ???
        }
        s"static constexpr ${jniTypeName(f.getType)} ${safeCppNames(f.getName)} = $rhs"
      }

    val prototypes = nativeMethods.map { m =>
      val self = if (Modifier.isStatic(m.getModifiers)) "jclass" else "jobject"
      ("JNIEnv *env" :: self :: m.getParameters.map(p => s"${jniTypeName(p.getType)} ${p.getName}").toList)
        .mkString(s"${jniTypeName(m.getReturnType)} ${safeCppNames(m.getName)}(", ", ", ")")
    }

    val jniName = jniClassName(cls)
    val clsName = cls.getSimpleName

    val header =
      s"""#pragma once
         |#include <jni.h>
         |#include <stdexcept>
         |namespace polyregion::generated::registry::$clsName {
         |${constants.map(c => s"$c;").mkString("\n")}
         |${prototypes.map(p => s"[[maybe_unused]] $p;").mkString("\n")}
         |
         |thread_local jclass clazz = nullptr;
         |
         |static void unregisterMethods(JNIEnv *env) {
         |  if (!clazz) return;
         |  if(env->UnregisterNatives(clazz) != 0){
         |    throw std::logic_error("UnregisterNatives returned non-zero for $jniName");
         |  }
         |  env->DeleteGlobalRef(clazz);
         |  clazz = nullptr;
         |}
         |
         |static void registerMethods(JNIEnv *env) {
         |  if (clazz) return;
         |  clazz = reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("$jniName")));
         |  const static JNINativeMethod methods[${registerEntries.length}] = {
         |${registerEntries.map("      " + _).mkString(",\n")}};
         |  if(env->RegisterNatives(clazz, methods, ${registerEntries.length}) != 0){
         |    throw std::logic_error("RegisterNatives returned non-zero for $jniName");
         |  }
         |}
         |
         |} // namespace polyregion::generated::registry::$clsName""".stripMargin

    clsName -> header

  }

}
