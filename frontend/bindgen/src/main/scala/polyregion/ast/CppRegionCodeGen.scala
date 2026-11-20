package polyregion.ast

import polyregion.ast.PolyAST as p

private[ast] object CppRegionCodeGen {
  import InterfaceCodeGenSupport.*
  import CppInterfaceCodeGen.{identifier as cppIdentifier, pascalCase, tpe}

  private def isCallable(arg: p.Arg): Boolean = arg.named.tpe.isInstanceOf[p.Type.Exec]

  private def parameter(arg: p.Arg): String =
    if (isCallable(arg)) s"${pascalCase(arg.named.symbol)} ${arg.named.symbol}"
    else s"${tpe(arg.named.tpe)} ${arg.named.symbol}"

  private def templates(decl: p.FunctionDecl): String = {
    val names = decl.tpeVars.map(variable => s"class ${variable.name}") ++
      decl.args.filter(isCallable).map(arg => s"class ${pascalCase(arg.named.symbol)}")
    if (names.isEmpty) "" else s"template <${names.mkString(", ")}>\n"
  }

  private def function(
      decl: p.FunctionDecl,
      name: String,
      leading: List[String],
      parameters: List[String],
      body: String
  ): String =
    s"""${templates(decl)}inline ${tpe(decl.rtn)} $name(${(leading ++ parameters).mkString(", ")}) {
       |  $body
       |}""".stripMargin

  private def forward(interfaceDef: p.Interface, decl: p.FunctionDecl, withContext: Boolean): String = {
    val callee    = s"::${interfaceDef.name.fqn.mkString("::")}::${decl.name.last}"
    val arguments = (if (withContext) List(s"$ContextParameter.handle()") else Nil) ++ decl.args.map(_.named.symbol)
    val call      = s"$callee(${arguments.mkString(", ")});"
    val body      = if (decl.rtn == p.Type.Unit0) call else s"return $call"
    function(
      decl,
      decl.name.last,
      if (withContext) List(s"const Context &$ContextParameter") else Nil,
      decl.args.map(parameter),
      body
    )
  }

  private def nd(decl: p.FunctionDecl, rank: Int, withContext: Boolean): String = {
    val dims     = extents(rank)
    val indices  = linearisedIndices(rank, LinearIndex).mkString(", ")
    val rest     = decl.args.tail
    val contextP = if (withContext) List(s"const Context &$ContextParameter") else Nil
    val extentP  = dims.map(dim => s"std::int64_t $dim")
    val restP    = rest.map(arg => parameter(arg))
    val total    = dims.tail.foldLeft(dims.head)((acc, dim) => s"detail::extent($acc, $dim)")
    val arguments = (if (withContext) List(ContextParameter) else Nil) ++ (total :: rest.map { arg =>
      if (isIndexCallable(arg.named.tpe)) s"[=](std::int64_t $LinearIndex) { return ${arg.named.symbol}($indices); }"
      else arg.named.symbol
    })
    val call = s"${decl.name.last}(${arguments.mkString(", ")});"
    function(decl, decl.name.last, contextP ++ extentP, restP, if (decl.rtn == p.Type.Unit0) call else s"return $call")
  }

  private def task(decl: p.FunctionDecl, withContext: Boolean): String = {
    val callable = decl.args.last
    val name     = pascalCase(callable.named.symbol)
    val leading  = if (withContext) s"const Context &$ContextParameter, " else ""
    val context  = if (withContext) s"$ContextParameter, " else ""
    s"""template <class $name>
       |inline void task(${leading}$name ${callable.named.symbol}) {
       |  ${decl.name.last}(${context}1, [=](std::int64_t) { return ${callable.named.symbol}(); });
       |}""".stripMargin
  }

  private val extentHelper =
    """namespace detail {
      |
      |inline std::int64_t extent(std::int64_t a, std::int64_t b) {
      |  std::int64_t product = 0;
      |  if (__builtin_mul_overflow(a, b, &product)) throw std::overflow_error("extent product overflows 64 bits");
      |  return product;
      |}
      |
      |} // namespace detail""".stripMargin

  private val contextClass =
    """class Context {
      |  void *handle_ = nullptr;
      |
      |public:
      |  struct Device {
      |    std::int32_t platform, device;
      |    std::string platformName, name;
      |  };
      |
      |  static std::vector<Device> devices() {
      |    std::int32_t platforms = 0;
      |    if (!polyrt_platform_count(&platforms)) throw std::runtime_error(polyrt_error_message());
      |    std::vector<Device> found;
      |    for (std::int32_t platform = 0; platform < platforms; ++platform) {
      |      PolyrtPlatformInfo platformInfo{};
      |      if (!polyrt_platform_info(platform, &platformInfo)) throw std::runtime_error(polyrt_error_message());
      |      for (std::int32_t device = 0; device < platformInfo.deviceCount; ++device) {
      |        PolyrtDeviceInfo deviceInfo{};
      |        if (!polyrt_device_info(platform, device, &deviceInfo)) throw std::runtime_error(polyrt_error_message());
      |        found.push_back({platform, device, platformInfo.name, deviceInfo.name});
      |      }
      |    }
      |    return found;
      |  }
      |
      |  Context(std::int32_t platform, std::int32_t device) {
      |    if (!polyrt_context_create(platform, device, &handle_)) throw std::runtime_error(polyrt_error_message());
      |  }
      |  explicit Context(const Device &device) : Context(device.platform, device.device) {}
      |  Context(const Context &) = delete;
      |  Context &operator=(const Context &) = delete;
      |  Context(Context &&other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
      |  Context &operator=(Context &&other) noexcept {
      |    std::swap(handle_, other.handle_);
      |    return *this;
      |  }
      |  ~Context() {
      |    if (handle_ && !polyrt_context_destroy(handle_)) std::fprintf(stderr, "polyregion::offload::Context: %s\\n", polyrt_error_message());
      |  }
      |
      |  void *handle() const { return handle_; }
      |};""".stripMargin

  def apply(interfaceDef: p.Interface, config: CppRegionConfig): String = {
    val decls = declarations(interfaceDef)
    validateRegionNames(decls)
    config.namespaceName.split("::").foreach(cppIdentifier(_, "namespace"))
    decls.foreach { decl =>
      cppIdentifier(decl.name.last, "function")
      decl.args.foreach(arg => cppIdentifier(arg.named.symbol, "parameter"))
    }
    val contexts = List(false, true)
    val body = (
      contexts.flatMap(withContext => decls.map(forward(interfaceDef, _, withContext)))
        ++ decls.filter(isIndexed).flatMap(decl => Ranks.flatMap(rank => contexts.map(nd(decl, rank, _))))
        ++ decls.filter(isTask).flatMap(decl => contexts.map(task(decl, _)))
    ).mkString("\n\n")
    val includes =
      (List("<cstdint>", "<cstdio>", "<stdexcept>", "<string>", "<utility>", "<vector>") ++ config.includes.map(i =>
        s"\"$i\""
      ))
        .map(i => s"#include $i")
    s"""#pragma once
       |
       |${includes.mkString("\n")}
       |
       |namespace ${config.namespaceName} {
       |
       |$extentHelper
       |
       |$contextClass
       |
       |$body
       |
       |} // namespace ${config.namespaceName}
       |""".stripMargin
  }
}
