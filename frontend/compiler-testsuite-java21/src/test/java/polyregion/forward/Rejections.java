package polyregion.forward;

final class Rejections {

  // XXX every pattern form reads the same once javac has bound the variable
  static final String PATTERN = "binding expression introduced by desugaring";

  // XXX names a unary operator rather than the pattern
  static final String STATEMENT_SWITCH_ARM = "Unsupported unary operator OTHER";

  static final String INSTANCEOF = "`instanceof`";

  private Rejections() {}
}
