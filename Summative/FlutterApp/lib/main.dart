import 'dart:convert';
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
      title: 'Depression Risk Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3949AB),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
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
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
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
                'Depression score: ${score.toStringAsFixed(4)} (0–1, higher = more risk)\n'
                'Classification: $risk\n'
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
        _resultMessage = 'Request failed: $e\n'
            'Check apiBaseUrl in main.dart and that the API is running.';
        _isError = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final bg = Color.lerp(theme.colorScheme.surfaceContainerLowest, theme.colorScheme.primaryContainer, 0.12)!;

    return Scaffold(
      backgroundColor: bg,
      appBar: AppBar(
        title: const Text('Depression risk predictor'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Enter each field below, then tap Predict. Sleep and diet values must match the training dataset wording. '
                'If the API returns a 503 about encoders.pkl, run POST /retrain once on the server with your CSV, then try again.',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 20),
              ..._sections.entries.map((entry) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: Card(
                    child: Padding(
                      padding: const EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            entry.key,
                            style: theme.textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w600,
                              color: theme.colorScheme.primary,
                            ),
                          ),
                          const SizedBox(height: 12),
                          ...entry.value.map((f) => Padding(
                                padding: const EdgeInsets.only(bottom: 12),
                                child: TextFormField(
                                  controller: _controllers[f.keyJson],
                                  decoration: InputDecoration(
                                    labelText: f.label,
                                    hintText: f.hint,
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
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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
                    : const Text('Predict'),
              ),
              if (_resultMessage != null) ...[
                const SizedBox(height: 20),
                Material(
                  color: _isError
                      ? theme.colorScheme.errorContainer
                      : theme.colorScheme.primaryContainer,
                  borderRadius: BorderRadius.circular(16),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(
                          _isError ? Icons.error_outline : Icons.check_circle_outline,
                          color: _isError
                              ? theme.colorScheme.onErrorContainer
                              : theme.colorScheme.onPrimaryContainer,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            _resultMessage!,
                            style: theme.textTheme.bodyLarge?.copyWith(
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
