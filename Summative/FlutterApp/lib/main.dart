// ignore_for_file: deprecated_member_use

import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

/// After deploying the API to Render, set this to your service URL (no trailing slash).
/// Swagger UI (for markers / grading): https://YOUR-SERVICE.onrender.com/docs
const String apiBaseUrl = 'https://mindease-n866.onrender.com';

void main() {
  runApp(const DepressionPredictorApp());
}

class DepressionPredictorApp extends StatelessWidget {
  const DepressionPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Depression risk predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3949AB),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          elevation: 0,
          scrolledUnderElevation: 0.5,
          surfaceTintColor: Colors.transparent,
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          shadowColor: Colors.black26,
          surfaceTintColor: Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: BorderSide(color: Colors.grey.shade200),
          ),
          color: Colors.white,
        ),
      ),
      home: const PredictionPage(),
    );
  }
}

class _FieldSpec {
  const _FieldSpec({
    required this.keyJson,
    required this.label,
    required this.hint,
    required this.keyboardType,
    this.isInt = false,
    this.isFloat = false,
    this.extraValidator,
  });

  final String keyJson;
  final String label;
  final String hint;
  final TextInputType keyboardType;
  final bool isInt;
  final bool isFloat;
  final String? Function(String?)? extraValidator;
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  String? _resultMessage;
  bool _isError = false;

  final _controllers = <String, TextEditingController>{};

  static final _sections = <String, List<_FieldSpec>>{
    'Profile': [
      _FieldSpec(
        keyJson: 'age',
        label: 'Age',
        hint: '15–60 (whole number)',
        keyboardType: TextInputType.number,
        isInt: true,
        extraValidator: (v) {
          final n = int.tryParse(v?.trim() ?? '');
          if (n == null) return 'Enter a valid integer';
          if (n < 15 || n > 60) return 'Must be between 15 and 60';
          return null;
        },
      ),
      _FieldSpec(
        keyJson: 'gender',
        label: 'Gender',
        hint: 'Male or Female',
        keyboardType: TextInputType.text,
        extraValidator: (v) {
          final s = v?.trim() ?? '';
          if (s != 'Male' && s != 'Female') {
            return 'Type exactly Male or Female';
          }
          return null;
        },
      ),
    ],
    'Academic': [
      _FieldSpec(
        keyJson: 'academic_pressure',
        label: 'Academic pressure',
        hint: '0–5',
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        isFloat: true,
        extraValidator: _rangeFloat(0, 5),
      ),
      _FieldSpec(
        keyJson: 'work_pressure',
        label: 'Work pressure',
        hint: '0–5',
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        isFloat: true,
        extraValidator: _rangeFloat(0, 5),
      ),
      _FieldSpec(
        keyJson: 'cgpa',
        label: 'CGPA',
        hint: '0–10',
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        isFloat: true,
        extraValidator: _rangeFloat(0, 10),
      ),
      _FieldSpec(
        keyJson: 'study_satisfaction',
        label: 'Study satisfaction',
        hint: '0–5',
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        isFloat: true,
        extraValidator: _rangeFloat(0, 5),
      ),
      _FieldSpec(
        keyJson: 'degree',
        label: 'Degree',
        hint: 'e.g. BSc, B.Pharm, M.Tech (must match CSV spelling)',
        keyboardType: TextInputType.text,
      ),
    ],
    'Lifestyle & health': [
      _FieldSpec(
        keyJson: 'sleep_duration',
        label: 'Sleep duration',
        hint: 'Must match dataset text exactly',
        keyboardType: TextInputType.text,
        extraValidator: (v) {
          const allowed = {
            '5-6 hours',
            '7-8 hours',
            'Less than 5 hours',
            'More than 8 hours',
            'Others',
          };
          final s = v?.trim() ?? '';
          if (!allowed.contains(s)) {
            return 'Use exactly: 5-6 hours | 7-8 hours | Less than 5 hours | More than 8 hours | Others';
          }
          return null;
        },
      ),
      _FieldSpec(
        keyJson: 'dietary_habits',
        label: 'Dietary habits',
        hint: 'Healthy | Moderate | Unhealthy | Others',
        keyboardType: TextInputType.text,
        extraValidator: (v) {
          const allowed = {'Healthy', 'Moderate', 'Unhealthy', 'Others'};
          final s = v?.trim() ?? '';
          if (!allowed.contains(s)) {
            return 'Use exactly: Healthy, Moderate, Unhealthy, or Others';
          }
          return null;
        },
      ),
      _FieldSpec(
        keyJson: 'work_study_hours',
        label: 'Work / study hours per day',
        hint: '0–24',
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        isFloat: true,
        extraValidator: _rangeFloat(0, 24),
      ),
      _FieldSpec(
        keyJson: 'financial_stress',
        label: 'Financial stress',
        hint: '1–5 (whole number)',
        keyboardType: TextInputType.number,
        isInt: true,
        extraValidator: (v) {
          final n = int.tryParse(v?.trim() ?? '');
          if (n == null) return 'Enter a valid integer';
          if (n < 1 || n > 5) return 'Must be between 1 and 5';
          return null;
        },
      ),
    ],
    'Risk factors': [
      _FieldSpec(
        keyJson: 'suicidal_thoughts',
        label: 'Ever had suicidal thoughts?',
        hint: 'Yes or No',
        keyboardType: TextInputType.text,
        extraValidator: (v) {
          final s = v?.trim() ?? '';
          if (s != 'Yes' && s != 'No') return 'Type exactly Yes or No';
          return null;
        },
      ),
      _FieldSpec(
        keyJson: 'family_history_mental_illness',
        label: 'Family history of mental illness',
        hint: 'Yes or No',
        keyboardType: TextInputType.text,
        extraValidator: (v) {
          final s = v?.trim() ?? '';
          if (s != 'Yes' && s != 'No') return 'Type exactly Yes or No';
          return null;
        },
      ),
    ],
  };

  static String? Function(String?) _rangeFloat(double min, double max) {
    return (v) {
      final x = double.tryParse(v?.trim() ?? '');
      if (x == null) return 'Enter a valid number';
      if (x < min || x > max) return 'Must be between $min and $max';
      return null;
    };
  }

  Iterable<_FieldSpec> get _allFields sync* {
    for (final list in _sections.values) {
      yield* list;
    }
  }

  @override
  void initState() {
    super.initState();
    for (final f in _allFields) {
      _controllers[f.keyJson] = TextEditingController(text: _defaultFor(f));
    }
  }

  String _defaultFor(_FieldSpec f) {
    switch (f.keyJson) {
      case 'age':
        return '29';
      case 'gender':
        return 'Male';
      case 'academic_pressure':
        return '2';
      case 'work_pressure':
        return '0';
      case 'cgpa':
        return '7.5';
      case 'study_satisfaction':
        return '4';
      case 'sleep_duration':
        return '5-6 hours';
      case 'dietary_habits':
        return 'Moderate';
      case 'degree':
        return 'BSc';
      case 'suicidal_thoughts':
        return 'No';
      case 'work_study_hours':
        return '6';
      case 'financial_stress':
        return '2';
      case 'family_history_mental_illness':
        return 'No';
      default:
        return '';
    }
  }

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  /// Short message for users; avoids raw ClientException / URI dumps.
  String _friendlyConnectionError(Object e) {
    if (e is TimeoutException) {
      return 'The server is slow to respond (common on free hosting). '
          'Wait a minute and tap Predict again.';
    }
    final t = e.toString();
    if (t.contains('Failed to fetch') ||
        t.contains('ClientException') ||
        t.contains('SocketException') ||
        t.contains('HandshakeException')) {
      if (kIsWeb) {
        return 'Browser blocked the API call (CORS). Redeploy Summative/API to Render '
            'so OPTIONS /predict allows your localhost origin, or run this app on '
            'Windows/Android (no CORS): flutter run -d windows';
      }
      return 'Unable to reach the prediction service. Check the API URL and network.';
    }
    return 'Something went wrong. Please try again.\n($t)';
  }

  String _formatApiError(dynamic decoded) {
    if (decoded is Map && decoded['detail'] != null) {
      final d = decoded['detail'];
      if (d is String) return d;
      if (d is List) {
        return d.map((e) {
          if (e is Map) {
            final loc = e['loc'];
            final msg = e['msg'] ?? e['type'];
            return '$loc: $msg';
          }
          return e.toString();
        }).join('\n');
      }
    }
    return decoded.toString();
  }

  Future<void> _predict() async {
    if (_formKey.currentState?.validate() != true) return;

    setState(() {
      _isLoading = true;
      _resultMessage = null;
      _isError = false;
    });

    try {
      final body = <String, dynamic>{};
      for (final f in _allFields) {
        final val = _controllers[f.keyJson]!.text.trim();
        if (val.isEmpty) {
          setState(() {
            _isLoading = false;
            _resultMessage = 'Missing value: ${f.label}';
            _isError = true;
          });
          return;
        }
        if (f.isInt) {
          body[f.keyJson] = int.parse(val);
        } else if (f.isFloat) {
          body[f.keyJson] = double.parse(val);
        } else {
          body[f.keyJson] = val;
        }
      }
      // API accepts financial_stress as float
      final fs = body['financial_stress'];
      if (fs is int) body['financial_stress'] = fs.toDouble();

      final url = Uri.parse('$apiBaseUrl/predict');
      if (kDebugMode) {
        debugPrint('POST $url keys=${body.keys.toList()}');
      }
      // Render free tier can take 30–60s to wake; default short timeouts fail first.
      final response = await http
          .post(
            url,
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(
            const Duration(seconds: 90),
            onTimeout: () => throw TimeoutException(
              'No response in 90s (server may be starting).',
            ),
          );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final score = data['depression_score'] as num?;
        final risk = data['risk_label'] as String?;
        final conf = data['confidence'] as String?;
        setState(() {
          _isLoading = false;
          if (score != null && risk != null && conf != null) {
            _resultMessage =
                'Score: ${score.toStringAsFixed(4)}\n'
                'Risk: $risk\n'
                'Confidence: $conf';
            _isError = false;
          } else {
            _resultMessage = 'Unexpected response: $data';
            _isError = true;
          }
        });
      } else {
        dynamic err;
        try {
          err = jsonDecode(response.body);
        } catch (_) {
          err = response.body;
        }
        setState(() {
          _isLoading = false;
          _resultMessage = 'Error (${response.statusCode}): ${_formatApiError(err)}';
          _isError = true;
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _resultMessage = _friendlyConnectionError(e);
        _isError = true;
      });
    }
  }

  static IconData _iconForSection(String title) {
    switch (title) {
      case 'Profile':
        return Icons.person_outline_rounded;
      case 'Academic':
        return Icons.menu_book_outlined;
      case 'Lifestyle & health':
        return Icons.nights_stay_outlined;
      case 'Risk factors':
        return Icons.favorite_border_rounded;
      default:
        return Icons.edit_note_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    // Soft lavender-blue wash (readable on all devices; cards stay white).
    final bg = Color.alphaBlend(
      theme.colorScheme.primaryContainer.withOpacity(0.44),
      theme.colorScheme.surfaceContainerLow,
    );

    return Scaffold(
      backgroundColor: bg,
      appBar: AppBar(
        title: Text(
          'Depression risk predictor',
          style: theme.textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.w600,
            letterSpacing: 0.3,
          ),
        ),
        backgroundColor: Color.alphaBlend(
          theme.colorScheme.primaryContainer.withOpacity(0.22),
          theme.colorScheme.surface,
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 12, 20, 36),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Padding(
                padding: const EdgeInsets.only(bottom: 18),
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: theme.colorScheme.surface.withOpacity(0.92),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: theme.colorScheme.outlineVariant.withOpacity(0.65),
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: theme.colorScheme.shadow.withOpacity(0.07),
                        blurRadius: 10,
                        offset: const Offset(0, 3),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(14, 14, 16, 14),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        DecoratedBox(
                          decoration: BoxDecoration(
                            color: theme.colorScheme.secondaryContainer.withOpacity(0.55),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Padding(
                            padding: const EdgeInsets.all(9),
                            child: Icon(
                              Icons.psychology_alt_outlined,
                              size: 22,
                              color: theme.colorScheme.onSecondaryContainer.withOpacity(0.85),
                            ),
                          ),
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Text(
                            'Enter your student and lifestyle details below to get a '
                            'depression risk estimate and confidence from the linear '
                            'regression model—use for coursework, not as medical advice.',
                            style: theme.textTheme.bodyMedium?.copyWith(
                              height: 1.5,
                              color: theme.colorScheme.onSurfaceVariant,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              ..._sections.entries.map((entry) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 14),
                  child: Card(
                    clipBehavior: Clip.antiAlias,
                    child: Padding(
                      padding: const EdgeInsets.fromLTRB(16, 14, 16, 8),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(
                                _iconForSection(entry.key),
                                size: 22,
                                color: theme.colorScheme.primary,
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  entry.key,
                                  style: theme.textTheme.titleMedium?.copyWith(
                                    fontWeight: FontWeight.w600,
                                    color: theme.colorScheme.onSurface,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 14),
                          ...entry.value.map((f) => Padding(
                                padding: const EdgeInsets.only(bottom: 14),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      f.label,
                                      style: theme.textTheme.titleSmall?.copyWith(
                                        fontWeight: FontWeight.w600,
                                        color: theme.colorScheme.onSurfaceVariant,
                                      ),
                                    ),
                                    const SizedBox(height: 8),
                                    TextFormField(
                                      controller: _controllers[f.keyJson],
                                      decoration: InputDecoration(
                                        hintText: f.hint,
                                        floatingLabelBehavior: FloatingLabelBehavior.never,
                                      ),
                                      keyboardType: f.keyboardType,
                                      inputFormatters: f.isInt
                                          ? [FilteringTextInputFormatter.digitsOnly]
                                          : f.isFloat
                                              ? [
                                                  FilteringTextInputFormatter.allow(RegExp(r'[0-9.]')),
                                                ]
                                              : null,
                                      validator: (v) {
                                        if (v == null || v.trim().isEmpty) {
                                          return 'Required';
                                        }
                                        return f.extraValidator?.call(v);
                                      },
                                    ),
                                  ],
                                ),
                              )),
                        ],
                      ),
                    ),
                  ),
                );
              }),
              FilledButton(
                onPressed: _isLoading ? null : _predict,
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
                ),
                child: _isLoading
                    ? SizedBox(
                        height: 22,
                        width: 22,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: theme.colorScheme.onPrimary,
                        ),
                      )
                    : Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.analytics_outlined, size: 22, color: theme.colorScheme.onPrimary),
                          const SizedBox(width: 10),
                          Text('Predict', style: TextStyle(color: theme.colorScheme.onPrimary)),
                        ],
                      ),
              ),
              if (_resultMessage != null) ...[
                const SizedBox(height: 20),
                Material(
                  color: _isError
                      ? theme.colorScheme.errorContainer.withOpacity(0.9)
                      : theme.colorScheme.primaryContainer.withOpacity(0.85),
                  borderRadius: BorderRadius.circular(16),
                  child: Padding(
                    padding: const EdgeInsets.all(18),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          _isError ? Icons.error_outline_rounded : Icons.check_circle_outline_rounded,
                          size: 26,
                          color: _isError
                              ? theme.colorScheme.onErrorContainer
                              : theme.colorScheme.onPrimaryContainer,
                        ),
                        const SizedBox(width: 14),
                        Expanded(
                          child: Text(
                            _resultMessage!,
                            style: theme.textTheme.bodyLarge?.copyWith(
                              height: 1.45,
                              color: _isError
                                  ? theme.colorScheme.onErrorContainer
                                  : theme.colorScheme.onPrimaryContainer,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
