
import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' as math;

import 'package:audioplayers/audioplayers.dart';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

// =================== CONFIG ===================

const kPrimaryColor = Color(0xFF4CAF50); // green accent
const kModelAsset = 'assets/visorv1.tflite';
const kScoreThreshold = 0.5;
const kIouThreshold = 0.45;

const int inputWidth = 640;
const int inputHeight = 640;

// 19 road marking classes
const List<String> kLabels = [
  "Bicycle_Lane",
  "Broken_and_Solid_Yellow_Lines",
  "Bus_Lane",
  "Cats_Eye",
  "Continuity_Lane",
  "Double_Solid_Yellow_or_White_Line",
  "Holding_Lane",
  "Loading_and_Unloading_Zone",
  "Motorcycle_Lane",
  "No_Loading_and_Unloading_Curb",
  "No_Parking_Curb",
  "Parking_Bay",
  "Pavement_Arrow",
  "Pedestrian_Lane",
  "Railroad_Crossing",
  "Rumble_Strips",
  "Single_Solid_Line",
  "Speed_Limit",
  "Transition_Line",
];

// =================== HELPERS ===================

extension ColorUtils on Color {
  Color darken([double amount = .1]) {
    final hsl = HSLColor.fromColor(this);
    final hslDark = hsl.withLightness((hsl.lightness - amount).clamp(0.0, 1.0));
    return hslDark.toColor();
  }
}

class Detection {
  final Rect rect;
  final String label;
  final double score;

  Detection({required this.rect, required this.label, required this.score});
}

// =================== MAIN APP ===================

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(VisorAIApp());
}

class VisorAIApp extends StatefulWidget {
  VisorAIApp({Key? key}) : super(key: key);

  @override
  State<VisorAIApp> createState() => _VisorAIAppState();
}

class _VisorAIAppState extends State<VisorAIApp> {
  ThemeMode _themeMode = ThemeMode.light;

  void _toggleTheme() {
    setState(() => _themeMode = _themeMode == ThemeMode.light ? ThemeMode.dark : ThemeMode.light);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'VisorAI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: kPrimaryColor),
        useMaterial3: true,
        brightness: Brightness.light,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: kPrimaryColor, brightness: Brightness.dark),
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      themeMode: _themeMode,
      debugShowCheckedModeBanner: false,
      home: LandingPage(onToggleTheme: _toggleTheme, isDark: _themeMode == ThemeMode.dark),
    );
  }
}

// =================== UI: LANDING ===================

class LandingPage extends StatelessWidget {
  final VoidCallback onToggleTheme;
  final bool isDark;

  const LandingPage({Key? key, required this.onToggleTheme, required this.isDark}) : super(key: key);

  Future<void> _handleDetectionTap(BuildContext context) async {
    final status = await Permission.camera.request();
    if (status.isGranted) {
      Navigator.of(context).push(MaterialPageRoute(builder: (_) => const DetectionPage()));
    } else {
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text("Permission Needed"),
          content: const Text("Camera access is required to use the Detection feature."),
          actions: [
            TextButton(onPressed: () => Navigator.of(context).pop(), child: const Text("OK")),
          ],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).colorScheme.surface,
      appBar: AppBar(
        backgroundColor: Theme.of(context).brightness == Brightness.dark
            ? Colors.grey[900]
            : Colors.grey[100],
        foregroundColor: isDark ? Colors.white : kPrimaryColor.darken(0.2),
        titleSpacing: 0,
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Image.asset('assets/Logo/logo_new.png', height: 90),
            const SizedBox(width: 1),
            Text(
              'VisorAI',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 20,
                color: isDark ? Colors.white : kPrimaryColor,
              ),
            ),
          ],),
        actions: [
          IconButton(
            tooltip: isDark ? 'Switch to light mode' : 'Switch to dark mode',
            onPressed: onToggleTheme,
            icon: Icon(isDark ? Icons.dark_mode : Icons.light_mode),
          ),
        ],
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Spacer(),

          Icon(Icons.traffic_rounded, size: 120, color: kPrimaryColor),
          const SizedBox(height: 16),
          Text(
            'Road Marking Detection',
            textAlign: TextAlign.center,
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.bold,
              color: Theme.of(context).colorScheme.onSurface,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Powered by TensorFlow Lite',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
          ),
          const Spacer(),

          // ================= Bottom Navigation Bar =================
          SafeArea(
            top: false,
            child: Stack(
              clipBehavior: Clip.none,
              alignment: Alignment.bottomCenter,
              children: [
                // Background bar
                Container(
                  margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  padding: const EdgeInsets.symmetric(horizontal: 42),
                  height: 80,
                  decoration: BoxDecoration(
                    color: Theme.of(context).brightness == Brightness.dark
                        ? Colors.grey[900]
                        : Colors.grey[100],
                    borderRadius: BorderRadius.circular(24),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.2),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      // Left icon + text (Info)
                      Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          IconButton(
                            icon: Icon(Icons.info_outline_rounded, size: 35,
                              color: Theme.of(context).brightness == Brightness.dark
                                  ? Colors.white
                                  : kPrimaryColor,
                            ),
                            onPressed: () => Navigator.of(context)
                                .push(MaterialPageRoute(builder: (_) => const HowItWorksPage())),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'Info',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: Theme.of(context).brightness == Brightness.dark
                                  ? Colors.white
                                  : Colors.black,
                            ),
                          ),
                        ],
                      ),

                      const Spacer(flex: 1),// increase flex to push further
                      const SizedBox(width: 20),

                      // Right icon + text (About)
                      Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          IconButton(
                            icon: Icon(Icons.help_outline_rounded, size: 35,
                              color: Theme.of(context).brightness == Brightness.dark
                                  ? Colors.white
                                  : kPrimaryColor,
                            ),
                            onPressed: () => Navigator.of(context)
                                .push(MaterialPageRoute(builder: (_) => const AboutPage())),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'About',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: Theme.of(context).brightness == Brightness.dark
                                  ? Colors.white
                                  : Colors.black,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

                // Floating center camera button with title "Detect"
                Positioned(
                  bottom: 28,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 100,
                        height: 95,
                        decoration: BoxDecoration(
                          color: kPrimaryColor,
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: Theme.of(context).brightness == Brightness.dark
                                ? Colors.grey[900]!
                                : Colors.grey[100]!,
                            width: 6,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.25),
                              blurRadius: 12,
                              offset: const Offset(0, 6),
                            ),
                            const BoxShadow(
                              color: Colors.white10,
                              blurRadius: 4,
                              offset: Offset(0, -2),
                            ),
                          ],
                        ),
                        child: IconButton(
                          icon: const Icon(Icons.camera_alt_outlined, color: Colors.white, size: 40),
                          onPressed: () => _handleDetectionTap(context),
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Detect',
                        style: TextStyle(
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                          color: Theme.of(context).brightness == Brightness.dark
                              ? Colors.white
                              : Colors.black,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// =================== UI: HOW IT WORKS & ABOUT ===================

class HowItWorksPage extends StatelessWidget {
  const HowItWorksPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Use triple-quoted strings for multiline paragraphs to avoid concatenation issues.
    const overview = '''
VisorAI is a compact on-device road marking detection app using a YOLO-style TensorFlow Lite model.
It is designed to run efficiently on mid-range Android devices and provides visual and audio feedback for detected road markings.
''';

    const captureFlow = '''
1. Tap Detection to open the camera and capture a single image.
2. The captured image is processed in-memory (no persistent temp files).
3. Preprocessing resizes the image to the model input (640×640) and normalizes pixel values.
''';

    const inference = '''
The TFLite model outputs a fixed grid of predictions. For each detection we:
- decode box coordinates (center x/y, width, height),
- compute per-class scores,
- filter by a confidence threshold, and
- apply Non-Maximum Suppression (NMS) to remove redundant overlapping boxes.
''';

    const feedback = '''
Detected boxes are overlaid on the captured image. The top scoring detection triggers an MP3 playback (bundled in assets/sounds) to aid users who rely on audio cues.
Labels are drawn above boxes with a contrasting rounded background for readability.
''';

    const performanceTips = '''
- Ensure the device has adequate lighting for better detection accuracy.
- Keep the camera steady while capturing.
- Use the provided Realme 6 optimization as a reference; models may require re-quantization for other devices.
''';

    const technicalDetails = '''
- Model input: 640×640 RGB normalized to [0,1].
- Output format: YOLO-like tensor decoded into [cx, cy, w, h, class_scores...].
- Thresholds: confidence (0.5) and IoU (0.45) used for NMS.
''';

    return Scaffold(
      appBar: AppBar(
        title: const Text('How it Works'),
        backgroundColor: kPrimaryColor,
        foregroundColor: Colors.white,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text('Overview', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          const Text(overview),
          const SizedBox(height: 16),

          const Text('Capture Flow', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          const Text(captureFlow),
          const SizedBox(height: 16),

          const Text('Inference & Post-processing', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          const Text(inference),
          const SizedBox(height: 16),

          const Text('Feedback & UX', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          const Text(feedback),
          const SizedBox(height: 16),

          const Text('Performance Tips', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          const Text(performanceTips),
          const SizedBox(height: 24),

          const Text('Technical Details', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
          const SizedBox(height: 8),
          const Text(technicalDetails),
        ],
      ),
    );
  }
}

class AboutPage extends StatelessWidget {
  const AboutPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final labelList = kLabels.map((l) => '• $l').join('\n');

    const credits = '''
- Model: visorv1.tflite (YOLO-style export).
- Audio cues: MP3 files located under assets/sounds/ named exactly as classes.
- Camera: Flutter camera plugin for image capture.
''';

    const license = '''
This app is a demonstration of on-device inference techniques. Verify local regulations before using in production scenarios; do not rely on it for safety-critical decisions.
''';

    return Scaffold(
      appBar: AppBar(
        title: const Text('About'),
        backgroundColor: kPrimaryColor,
        foregroundColor: Colors.white,
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text("VisorAI", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
          const SizedBox(height: 8),
          const Text('On-device road marking detection powered by TensorFlow Lite.'),
          const SizedBox(height: 16),

          const Text('Goals', style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text(
            'Provide a lightweight, low-latency detection experience that can be used for assistive applications, data collection, or research on road markings in constrained mobile environments.',
          ),
          const SizedBox(height: 16),

          const Text('Included Classes', style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(labelList),
          const SizedBox(height: 16),

          const Text('Credits & Assets', style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text(credits),
          const SizedBox(height: 24),

          const Text('License & Usage', style: TextStyle(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          const Text(license),
        ],
      ),
    );
  }
}

// =================== UI: DETECTION ===================

class DetectionPage extends StatefulWidget {
  const DetectionPage({Key? key}) : super(key: key);

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage> {
  CameraController? _cameraController;
  bool _loading = true;
  final _engine = _InferenceEngine();
  final _player = AudioPlayer();

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    try {
      final cameras = await availableCameras();
      _cameraController = CameraController(cameras.first, ResolutionPreset.medium, enableAudio: false);
      await _cameraController!.initialize();
      await _engine.loadModel();
    } catch (e) {
      debugPrint("Init error: $e");
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _captureAndDetect() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;

    try {
      final file = await _cameraController!.takePicture();
      final bytes = await File(file.path).readAsBytes();

      final codecOrig = await ui.instantiateImageCodec(bytes);
      final frameOrig = await codecOrig.getNextFrame();
      final origImage = frameOrig.image;

      final detections = await _engine.run(bytes, originalW: origImage.width, originalH: origImage.height);

      if (detections.isNotEmpty) {
        final top = detections.first;
        try {
          await _player.stop();
          await _player.play(AssetSource("sounds/${top.label}.mp3"));
        } catch (e) {
          debugPrint("Audio error: $e");
        }
      }

      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();

      if (!mounted) return;
      Navigator.push(context, MaterialPageRoute(builder: (_) => PreviewPage(image: frame.image, detections: detections)));
    } catch (e) {
      debugPrint("Capture error: $e");
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _engine.close();
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Detection"),
        backgroundColor: kPrimaryColor,
        foregroundColor: Colors.white,
      ),
      body: Stack(
        children: [
          if (_cameraController != null) CameraPreview(_cameraController!),
          Positioned(
            bottom: 24,
            left: 24,
            right: 24,
            child: ElevatedButton.icon(
              onPressed: _captureAndDetect,
              icon: const Icon(Icons.camera_alt),
              label: const Text("Capture & Detect"),
              style: ElevatedButton.styleFrom(
                backgroundColor: kPrimaryColor,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 14),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// =================== UI: PREVIEW ===================

class PreviewPage extends StatelessWidget {
  final ui.Image image;
  final List<Detection> detections;

  const PreviewPage({Key? key, required this.image, required this.detections}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Detection Result"),
        backgroundColor: kPrimaryColor,
        foregroundColor: Colors.white,
      ),
      body: Center(
        child: FittedBox(
          child: SizedBox(
            width: image.width.toDouble(),
            height: image.height.toDouble(),
            child: CustomPaint(painter: DetectionPainter(image: image, detections: detections)),
          ),
        ),
      ),
    );
  }
}

class DetectionPainter extends CustomPainter {
  final ui.Image image;
  final List<Detection> detections;

  DetectionPainter({required this.image, required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    final src = Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble());
    final dst = Rect.fromLTWH(0, 0, size.width, size.height);
    canvas.drawImageRect(image, src, dst, Paint());

    final scaleX = size.width / image.width;
    final scaleY = size.height / image.height;

    final boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.red;

    for (final det in detections) {
      final r = Rect.fromLTRB(
        det.rect.left * scaleX,
        det.rect.top * scaleY,
        det.rect.right * scaleX,
        det.rect.bottom * scaleY,
      );

      // Draw bounding box
      canvas.drawRect(r, boxPaint);

      // Prepare label text with score
      final labelText = det.label;
      final textSpan = TextSpan(
        text: labelText,
        style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
      );

      final tp = TextPainter(text: textSpan, textDirection: TextDirection.ltr);
      tp.layout();

      // Position the label above the box if there is space, otherwise below
      double textTop = r.top - tp.height - 6;
      if (textTop < 0) textTop = r.top + 4;

      final bgRect = Rect.fromLTWH(r.left, textTop, tp.width + 8, tp.height + 4);
      final bgRRect = RRect.fromRectAndRadius(bgRect, const Radius.circular(6));
      final bgPaint = Paint()..color = kPrimaryColor.withOpacity(0.9);
      canvas.drawRRect(bgRRect, bgPaint);

      // paint the text
      tp.paint(canvas, Offset(r.left + 4, textTop + 2));
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter oldDelegate) => true;
}

// =================== INFERENCE ENGINE ===================

class _InferenceEngine {
  tfl.Interpreter? _interpreter;

  Future<void> loadModel() async {
    _interpreter = await tfl.Interpreter.fromAsset(kModelAsset);
  }

  Future<List<Detection>> run(Uint8List bytes, {required int originalW, required int originalH}) async {
    if (_interpreter == null) return [];

    final codec = await ui.instantiateImageCodec(bytes, targetWidth: inputWidth, targetHeight: inputHeight);
    final frame = await codec.getNextFrame();
    final img = frame.image;
    final imgBytes = await img.toByteData(format: ui.ImageByteFormat.rawRgba);

    if (imgBytes == null) return [];

    // Build input as [1, inputHeight, inputWidth, 3]
    final input = List.generate(1, (_) => List.generate(inputHeight, (y) => List.generate(inputWidth, (x) {
      final i = (y * inputWidth + x) * 4;
      return [
        imgBytes.getUint8(i) / 255.0,
        imgBytes.getUint8(i + 1) / 255.0,
        imgBytes.getUint8(i + 2) / 255.0,
      ];
    })));

    // ✅ Expected output shape [1, 23, 8400]
    final output = List.generate(1, (_) => List.generate(23, (_) => List.filled(8400, 0.0)));

    _interpreter!.run(input, output);

    final dets = _parseDetections(output[0], imgW: inputWidth, imgH: inputHeight);

    // Scale back to original image size
    final scaleX = originalW / inputWidth;
    final scaleY = originalH / inputHeight;

    return _nonMaxSuppression(dets.map((d) {
      return Detection(
        rect: Rect.fromLTRB(d.rect.left * scaleX, d.rect.top * scaleY, d.rect.right * scaleX, d.rect.bottom * scaleY),
        label: d.label,
        score: d.score,
      );
    }).toList());
  }

  void close() => _interpreter?.close();

  List<Detection> _parseDetections(List raw, {required int imgW, required int imgH}) {
    final out = <Detection>[];

    // raw is expected as List of 23 lists each of length 8400
    for (var i = 0; i < 8400; i++) {
      final cx = (raw[0][i] as num).toDouble();
      final cy = (raw[1][i] as num).toDouble();
      final w = (raw[2][i] as num).toDouble();
      final h = (raw[3][i] as num).toDouble();

      // Ensure scores are <double> (class scores start at index 4 up to 22 inclusive -> 19 classes)
      final scores = <double>[];
      for (var c = 4; c < 23; c++) {
        scores.add((raw[c][i] as num).toDouble());
      }

      final maxVal = scores.reduce(math.max);
      final maxIdx = scores.indexOf(maxVal);

      if (maxVal < kScoreThreshold) continue;

      out.add(Detection(
        rect: Rect.fromLTRB((cx - w / 2) * imgW, (cy - h / 2) * imgH, (cx + w / 2) * imgW, (cy + h / 2) * imgH),
        label: kLabels[maxIdx],
        score: maxVal,
      ));
    }

    return out;
  }

  List<Detection> _nonMaxSuppression(List<Detection> dets) {
    dets.sort((a, b) => b.score.compareTo(a.score));
    final kept = <Detection>[];
    for (final d in dets) {
      if (kept.every((k) => _iou(k.rect, d.rect) < kIouThreshold)) {
        kept.add(d);
      }
    }
    return kept;
  }

  double _iou(Rect a, Rect b) {
    final inter = Rect.fromLTRB(
      math.max(a.left, b.left),
      math.max(a.top, b.top),
      math.min(a.right, b.right),
      math.min(a.bottom, b.bottom),
    );
    if (inter.width <= 0 || inter.height <= 0) return 0.0;
    final interArea = inter.width * inter.height;
    final union = a.width * a.height + b.width * b.height - interArea;
    return interArea / union;
  }
}
