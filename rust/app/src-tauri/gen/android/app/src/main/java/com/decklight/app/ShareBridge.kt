package com.decklight.app

import android.content.Intent
import android.net.Uri
import android.util.Base64
import android.util.Log
import android.webkit.JavascriptInterface
import androidx.core.content.FileProvider
import java.io.File
import java.io.FileOutputStream
import org.json.JSONArray

// The deck builder's Export sheet calls window.DecklightShare.shareYdk(name, text) when the bridge exists. A WebView
// Blob download does nothing, so this writes the .ydk to cache/share/ and opens the system share sheet instead.
// Games > Export (web/features/games/export.js) sends any files the same way: reset(), then each file in base64
// chunks (begin, append...), then send(names). Each call returns "" or the error for the screen to show.
class ShareBridge(private val activity: MainActivity) {
  private val dir get() = File(activity.cacheDir, "share")

  private fun safe(name: String): String {
    val cleaned = name.replace(Regex("[^\\p{L}\\p{N} .,_()-]"), "").trim().trimStart('.')
    return cleaned.take(120).ifEmpty { "export" }
  }

  private fun attempt(what: String, f: () -> Unit): String = try {
    f(); ""
  } catch (e: Exception) {
    Log.w("decklight", "$what failed", e)
    "$what failed: ${e.message ?: e.javaClass.simpleName}"
  }

  @JavascriptInterface
  fun reset(): String = attempt("clearing the share folder") {
    dir.listFiles()?.forEach { it.delete() }
    if (!dir.isDirectory && !dir.mkdirs()) throw IllegalStateException("cannot create ${dir.path}")
  }

  @JavascriptInterface
  fun begin(name: String): String = attempt("writing $name") { File(dir, safe(name)).writeBytes(ByteArray(0)) }

  @JavascriptInterface
  fun append(name: String, b64: String): String = attempt("writing $name") {
    FileOutputStream(File(dir, safe(name)), true).use { it.write(Base64.decode(b64, Base64.DEFAULT)) }
  }

  @JavascriptInterface
  fun send(namesJson: String, mime: String): String = attempt("opening the share sheet") {
    val names = JSONArray(namesJson)
    val uris = ArrayList<Uri>()
    for (i in 0 until names.length()) {
      val f = File(dir, safe(names.getString(i)))
      if (!f.isFile) throw IllegalStateException("${f.name} was not written")
      uris.add(FileProvider.getUriForFile(activity, "${activity.packageName}.fileprovider", f))
    }
    if (uris.isEmpty()) throw IllegalStateException("no files to share")
    val send = if (uris.size == 1) {
      Intent(Intent.ACTION_SEND).apply { putExtra(Intent.EXTRA_STREAM, uris[0]) }
    } else {
      Intent(Intent.ACTION_SEND_MULTIPLE).apply { putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris) }
    }
    send.type = mime.ifEmpty { "application/octet-stream" }
    send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    val title = if (uris.size == 1) names.getString(0) else "${uris.size} files"
    activity.runOnUiThread { activity.startActivity(Intent.createChooser(send, title)) }
  }

  @JavascriptInterface
  fun shareYdk(fileName: String, text: String) {
    try {
      val cleaned = fileName.replace(Regex("[^A-Za-z0-9 ._-]"), "").trim()
      var base = if (cleaned.lowercase().endsWith(".ydk")) cleaned.dropLast(4) else cleaned
      base = base.trim().take(56).trim().trimEnd('.', ' ')
      if (base.isEmpty()) base = "deck"
      val name = "$base.ydk"

      val dir = File(activity.cacheDir, "share")
      if (!dir.isDirectory && !dir.mkdirs()) throw IllegalStateException("cannot create ${dir.path}")
      val file = File(dir, name)
      file.writeText(text)

      val uri = FileProvider.getUriForFile(activity, "${activity.packageName}.fileprovider", file)
      val send = Intent(Intent.ACTION_SEND).apply {
        type = "application/octet-stream"
        putExtra(Intent.EXTRA_STREAM, uri)
        putExtra(Intent.EXTRA_TEXT, text)
        putExtra(Intent.EXTRA_SUBJECT, name)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
      }
      activity.runOnUiThread { activity.startActivity(Intent.createChooser(send, name)) }
    } catch (e: Exception) {
      Log.w("decklight", "shareYdk failed", e)
    }
  }
}
